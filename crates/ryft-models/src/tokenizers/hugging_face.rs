//! [`HuggingFaceTokenizer`], which implements [`Tokenizer`] for the `tokenizer.json` files of Hugging Face model
//! releases.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

use fancy_regex::Regex;
use serde_json::{Map, Value};

use crate::tokenizers::{SpecialTokenHandling, TokenId, Tokenizer, TokenizerError};

// TODO(eaplatanios): Review this module.

/// [`Tokenizer`] defined by a Hugging Face `tokenizer.json` file, which it reads natively.
///
/// It supports byte level BPE tokenizers, i.e., definitions with the following components.
///
///   - `normalizer`: `null`.
///   - `pre_tokenizer`: either a lone `ByteLevel` pre-tokenizer, or a `Sequence` of `Split` pre-tokenizers followed by
///     one `ByteLevel` pre-tokenizer. Each `Split` pre-tokenizer has a `Regex` pattern, the `Isolated` behavior, and
///     no inversion, so every match of its pattern and every gap between two matches becomes its own piece. The
///     patterns are compiled with [`fancy_regex`], which supports the lookarounds of common pretokenization
///     patterns. The `ByteLevel` pre-tokenizer has neither `add_prefix_space` nor `use_regex` enabled. It maps every
///     UTF-8 byte of a piece to one character of the GPT-2 byte level alphabet, in which the vocabulary is spelled.
///   - `model`: `BPE` without dropout, unknown tokens, subword prefixes or suffixes, or byte fallback. Each piece is
///     merged by repeatedly merging the adjacent pair of tokens with the lowest merge rank, and the leftmost such pair
///     among equal ranks. When `ignore_merges` is enabled, a piece whose whole byte level string is a vocabulary token
///     becomes that token without merging. Merges may be spelled as two element arrays or as strings that separate
///     their two tokens with a space.
///   - `decoder`: `ByteLevel`, which maps the byte level characters of tokens back to bytes and decodes those bytes
///     as UTF-8, replacing invalid sequences with `U+FFFD` like [`String::from_utf8_lossy`].
///   - `post_processor`: `null` or `ByteLevel`, which only adjusts offsets and therefore does not affect tokens.
///   - `truncation` and `padding`: `null`.
///   - `added_tokens`: tokens without `lstrip`, `rstrip`, or `single_word`, and without `normalized` when they are
///     special. Their contents are matched in the text before pretokenization, taking the leftmost match and the
///     longest content among the matches at one position, and each match becomes its token directly. With
///     [`SpecialTokenHandling::Literal`], only the contents of the added tokens that are not special are matched.
///     [`Tokenizer::decode`] appends the contents of added tokens verbatim and drops special ones when asked to.
///
/// Loading any other definition fails with a [`TokenizerError::UnsupportedDefinition`] that names the unsupported
/// component or option.
#[derive(Clone, Debug)]
pub struct HuggingFaceTokenizer {
    /// Patterns of the `Split` pre-tokenizers, applied in order before the byte level mapping.
    split_patterns: Vec<Regex>,

    /// Identifiers of the model tokens, keyed by their byte level strings.
    vocabulary: HashMap<String, TokenId>,

    /// Identifiers of the added tokens, keyed by their contents.
    added_token_ids: HashMap<String, TokenId>,

    /// Entries of all tokens indexed by their identifiers, with [`None`] for identifiers that no token uses.
    entries: Vec<Option<VocabularyEntry>>,

    /// Identifier of the token of each single byte, indexed by that byte, or [`None`] if the vocabulary has no such
    /// token.
    byte_tokens: [Option<TokenId>; 256],

    /// BPE merges keyed by the pair of token identifiers that they merge.
    merges: HashMap<(TokenId, TokenId), Merge>,

    /// Whether a piece whose byte level string is a vocabulary token becomes that token without merging.
    ignore_merges: bool,

    /// Pattern that matches the contents of all added tokens, or [`None`] if there are no added tokens.
    added_token_pattern: Option<Regex>,

    /// Pattern that matches the contents of the added tokens that are not special, or [`None`] if there are none.
    non_special_added_token_pattern: Option<Regex>,
}

impl HuggingFaceTokenizer {
    /// Creates a [`HuggingFaceTokenizer`] from the contents of a `tokenizer.json` file.
    pub fn from_json(json: &str) -> Result<Self, TokenizerError> {
        let definition = serde_json::from_str::<Value>(json)
            .map_err(|error| TokenizerError::InvalidDefinition { message: error.to_string() })?;
        Self::from_definition(definition)
    }

    /// Creates a [`HuggingFaceTokenizer`] from the `tokenizer.json` file at `path`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path)
            .map_err(|error| TokenizerError::Io { message: format!("`{}`: {error}", path.display()) })?;
        Self::from_json(&json)
    }

    /// Creates a [`HuggingFaceTokenizer`] from a parsed `tokenizer.json` definition.
    fn from_definition(definition: Value) -> Result<Self, TokenizerError> {
        let Value::Object(mut definition) = definition else {
            return Err(TokenizerError::InvalidDefinition {
                message: "the tokenizer definition must be a JSON object".to_string(),
            });
        };
        validate_option(&definition, "normalizer", &Value::Null, &Value::Null, "normalizer")?;
        validate_option(&definition, "truncation", &Value::Null, &Value::Null, "truncation")?;
        validate_option(&definition, "padding", &Value::Null, &Value::Null, "padding")?;
        let split_patterns = parse_pre_tokenizer(definition.get("pre_tokenizer").unwrap_or(&Value::Null))?;
        match definition.get("post_processor").unwrap_or(&Value::Null) {
            Value::Null => {}
            post_processor => match component(post_processor, "post_processor")? {
                (_, "ByteLevel") => {}
                (_, r#type) => return Err(unsupported_type("post_processor", r#type)),
            },
        }
        match definition.get("decoder").unwrap_or(&Value::Null) {
            Value::Null => {
                return Err(TokenizerError::UnsupportedDefinition {
                    message: "`decoder` must be a `ByteLevel` decoder".to_string(),
                });
            }
            decoder => match component(decoder, "decoder")? {
                (_, "ByteLevel") => {}
                (_, r#type) => return Err(unsupported_type("decoder", r#type)),
            },
        }

        let mut model = definition.remove("model").unwrap_or(Value::Null);
        match component(&model, "model")? {
            (model, "BPE") => {
                for key in ["dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix"] {
                    validate_option(model, key, &Value::Null, &Value::Null, &format!("model.{key}"))?;
                }
                for key in ["fuse_unk", "byte_fallback"] {
                    validate_option(model, key, &Value::Bool(false), &Value::Bool(false), &format!("model.{key}"))?;
                }
            }
            (_, r#type) => return Err(unsupported_type("model", r#type)),
        }
        let ignore_merges = match model.get("ignore_merges") {
            None => false,
            Some(Value::Bool(ignore_merges)) => *ignore_merges,
            Some(_) => {
                return Err(TokenizerError::InvalidDefinition {
                    message: "`model.ignore_merges` must be a boolean".to_string(),
                });
            }
        };
        let (vocabulary, mut entries) = parse_vocabulary(model.get_mut("vocab").map(Value::take))?;
        let merges = parse_merges(model.get("merges"), &vocabulary)?;
        let byte_tokens = std::array::from_fn(|byte| {
            let mut character = [0u8; 4];
            vocabulary.get(BYTE_CHARACTERS[byte].encode_utf8(&mut character) as &str).copied()
        });

        let mut added_token_ids = HashMap::new();
        let mut added_token_contents = Vec::new();
        let mut non_special_added_token_contents = Vec::new();
        let added_tokens = match definition.get("added_tokens") {
            None => &Vec::new(),
            Some(Value::Array(added_tokens)) => added_tokens,
            Some(_) => {
                return Err(TokenizerError::InvalidDefinition {
                    message: "`added_tokens` must be an array".to_string(),
                });
            }
        };
        for (index, added_token) in added_tokens.iter().enumerate() {
            let path = format!("added_tokens[{index}]");
            let Value::Object(added_token) = added_token else {
                return Err(TokenizerError::InvalidDefinition { message: format!("`{path}` must be an object") });
            };
            let id = added_token.get("id").and_then(token_id_from_value).ok_or_else(|| {
                TokenizerError::InvalidDefinition { message: format!("`{path}.id` must be a 32-bit unsigned integer") }
            })?;
            let content = match added_token.get("content") {
                Some(Value::String(content)) if !content.is_empty() => content.clone(),
                _ => {
                    return Err(TokenizerError::InvalidDefinition {
                        message: format!("`{path}.content` must be a non-empty string"),
                    });
                }
            };
            let special = match added_token.get("special") {
                None => false,
                Some(Value::Bool(special)) => *special,
                Some(_) => {
                    return Err(TokenizerError::InvalidDefinition {
                        message: format!("`{path}.special` must be a boolean"),
                    });
                }
            };
            for key in ["single_word", "lstrip", "rstrip"] {
                validate_option(added_token, key, &Value::Bool(false), &Value::Bool(false), &format!("{path}.{key}"))?;
            }
            if special {
                // Added tokens that are not special are normalized by default, but special ones are not.
                let normalized_path = format!("{path}.normalized");
                validate_option(added_token, "normalized", &Value::Bool(false), &Value::Bool(false), &normalized_path)?;
            }
            let index = id as usize;
            if entries.len() <= index {
                entries.resize_with(index + 1, || None);
            }
            match &entries[index] {
                Some(entry) if entry.added => {
                    return Err(TokenizerError::InvalidDefinition {
                        message: format!("`{path}` reuses the identifier {id} of added token `{}`", entry.content),
                    });
                }
                Some(entry) if entry.content != content => {
                    return Err(TokenizerError::InvalidDefinition {
                        message: format!("`{path}` reuses the identifier {id} of model token `{}`", entry.content),
                    });
                }
                _ => {}
            }
            if added_token_ids.insert(content.clone(), id).is_some() {
                return Err(TokenizerError::InvalidDefinition {
                    message: format!("`{path}` repeats the content `{content}` of an earlier added token"),
                });
            }
            if !special {
                non_special_added_token_contents.push(content.clone());
            }
            added_token_contents.push(content.clone());
            entries[index] = Some(VocabularyEntry { content, added: true, special });
        }

        Ok(Self {
            split_patterns,
            vocabulary,
            added_token_ids,
            entries,
            byte_tokens,
            merges,
            ignore_merges,
            added_token_pattern: added_token_pattern(added_token_contents)?,
            non_special_added_token_pattern: added_token_pattern(non_special_added_token_contents)?,
        })
    }

    /// Appends the tokens of `segment`, a span of text that contains no added token matches, to `tokens`.
    fn encode_segment(&self, segment: &str, tokens: &mut Vec<TokenId>) -> Result<(), TokenizerError> {
        let mut pieces = vec![segment];
        for pattern in &self.split_patterns {
            let mut split_pieces = Vec::with_capacity(pieces.len());
            for piece in pieces {
                let mut start = 0;
                for found in pattern.find_iter(piece) {
                    let found = found.map_err(|error| TokenizerError::Encoding {
                        message: format!("the `Split` pre-tokenizer pattern failed: {error}"),
                    })?;
                    split_pieces.push(&piece[start..found.start()]);
                    split_pieces.push(found.as_str());
                    start = found.end();
                }
                split_pieces.push(&piece[start..]);
            }
            split_pieces.retain(|piece| !piece.is_empty());
            pieces = split_pieces;
        }
        for piece in pieces {
            self.encode_piece(piece, tokens)?;
        }
        Ok(())
    }

    /// Appends the BPE tokens of `piece`, one output of pretokenization, to `tokens`.
    fn encode_piece(&self, piece: &str, tokens: &mut Vec<TokenId>) -> Result<(), TokenizerError> {
        if piece.is_empty() {
            return Ok(());
        }
        if self.ignore_merges {
            let byte_level_piece = piece.bytes().map(|byte| BYTE_CHARACTERS[byte as usize]).collect::<String>();
            if let Some(id) = self.vocabulary.get(&byte_level_piece) {
                tokens.push(*id);
                return Ok(());
            }
        }
        let mut symbols = piece
            .bytes()
            .map(|byte| {
                self.byte_tokens[byte as usize].ok_or_else(|| TokenizerError::Encoding {
                    message: format!("the vocabulary has no token for the byte `{byte:#04x}`"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // The symbols form a doubly linked list so that merging two of them takes constant time, and a heap orders
        // the candidate merges by rank and then by position. Entries whose pair changed after they were queued are
        // stale and skipped, which is detected by looking the current pair up again.
        let count = symbols.len();
        let mut previous = (0..count).map(|position| position.checked_sub(1)).collect::<Vec<_>>();
        let mut next = (0..count).map(|position| Some(position + 1).filter(|next| *next < count)).collect::<Vec<_>>();
        let mut merged = vec![false; count];
        let mut queue = BinaryHeap::new();
        for position in 1..count {
            if let Some(merge) = self.merges.get(&(symbols[position - 1], symbols[position])) {
                queue.push(Reverse((merge.rank, position - 1, merge.id)));
            }
        }
        while let Some(Reverse((rank, position, id))) = queue.pop() {
            if merged[position] {
                continue;
            }
            let Some(right) = next[position] else {
                continue;
            };
            match self.merges.get(&(symbols[position], symbols[right])) {
                Some(merge) if merge.rank == rank && merge.id == id => {}
                _ => continue,
            }
            symbols[position] = id;
            merged[right] = true;
            next[position] = next[right];
            if let Some(following) = next[right] {
                previous[following] = Some(position);
            }
            if let Some(left) = previous[position]
                && let Some(merge) = self.merges.get(&(symbols[left], symbols[position]))
            {
                queue.push(Reverse((merge.rank, left, merge.id)));
            }
            if let Some(following) = next[position]
                && let Some(merge) = self.merges.get(&(symbols[position], symbols[following]))
            {
                queue.push(Reverse((merge.rank, position, merge.id)));
            }
        }

        // The first symbol is never merged into its left neighbor, so the list always starts at position 0.
        let mut position = Some(0);
        while let Some(current) = position {
            tokens.push(symbols[current]);
            position = next[current];
        }
        Ok(())
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn encode(&self, text: &str, special_tokens: SpecialTokenHandling) -> Result<Vec<TokenId>, TokenizerError> {
        let pattern = match special_tokens {
            SpecialTokenHandling::Parse => self.added_token_pattern.as_ref(),
            SpecialTokenHandling::Literal => self.non_special_added_token_pattern.as_ref(),
        };
        let mut tokens = Vec::new();
        let mut start = 0;
        if let Some(pattern) = pattern {
            for found in pattern.find_iter(text) {
                let found = found.map_err(|error| TokenizerError::Encoding {
                    message: format!("the added token pattern failed: {error}"),
                })?;
                self.encode_segment(&text[start..found.start()], &mut tokens)?;
                tokens.push(self.added_token_ids[found.as_str()]);
                start = found.end();
            }
        }
        self.encode_segment(&text[start..], &mut tokens)?;
        Ok(tokens)
    }

    fn decode(&self, tokens: &[TokenId], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        // Model tokens contribute bytes that are decoded together, so that multibyte characters may span tokens, and
        // added tokens contribute their contents verbatim.
        let mut text = String::new();
        let mut bytes = Vec::new();
        for id in tokens {
            let Some(entry) = self.entries.get(*id as usize).and_then(Option::as_ref) else {
                return Err(TokenizerError::Decoding { message: format!("unknown token identifier {id}") });
            };
            if !entry.added {
                // Like the reference `ByteLevel` decoder, a token with characters outside the byte level alphabet
                // contributes its UTF-8 bytes unchanged.
                let token_bytes = entry.content.chars().map(byte_level_byte).collect::<Option<Vec<_>>>();
                bytes.extend(token_bytes.as_deref().unwrap_or(entry.content.as_bytes()));
            } else if !(skip_special_tokens && entry.special) {
                text.push_str(&String::from_utf8_lossy(&bytes));
                bytes.clear();
                text.push_str(&entry.content);
            }
        }
        text.push_str(&String::from_utf8_lossy(&bytes));
        Ok(text)
    }

    fn vocabulary_size(&self) -> usize {
        self.entries.len()
    }

    fn token_id(&self, token: &str) -> Option<TokenId> {
        self.added_token_ids.get(token).or_else(|| self.vocabulary.get(token)).copied()
    }

    fn token(&self, id: TokenId) -> Option<&str> {
        self.entries.get(id as usize).and_then(Option::as_ref).map(|entry| entry.content.as_str())
    }
}

/// Token of a [`HuggingFaceTokenizer`] vocabulary.
#[derive(Clone, Debug)]
struct VocabularyEntry {
    /// Byte level string of a model token, or the content of an added token.
    content: String,

    /// Whether this token is an added token.
    added: bool,

    /// Whether this token is a special added token.
    special: bool,
}

/// BPE merge of a pair of adjacent tokens.
#[derive(Copy, Clone, Debug)]
struct Merge {
    /// Priority of this merge, where lower ranks merge first.
    rank: usize,

    /// Identifier of the token that this merge produces.
    id: TokenId,
}

/// Characters of the GPT-2 byte level alphabet, indexed by the byte that they represent.
const BYTE_CHARACTERS: [char; 256] = byte_characters();

/// Bytes represented by the characters of the GPT-2 byte level alphabet, indexed by character code point.
const CHARACTER_BYTES: [Option<u8>; 324] = character_bytes();

/// Returns the GPT-2 byte level alphabet (i.e., the `bytes_to_unicode` table of GPT-2). The bytes `0x21..=0x7e`,
/// `0xa1..=0xac`, and `0xae..=0xff` represent the Latin-1 characters with the same code points, and the remaining 68
/// bytes map to the characters from `U+0100` onward in increasing byte order.
const fn byte_characters() -> [char; 256] {
    let mut characters = ['\0'; 256];
    let mut next_code_point = 256u32;
    let mut byte = 0usize;
    while byte < 256 {
        let code_point = if matches!(byte, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff) {
            byte as u32
        } else {
            next_code_point += 1;
            next_code_point - 1
        };
        characters[byte] = char::from_u32(code_point).unwrap();
        byte += 1;
    }
    characters
}

/// Returns the inverse of [`BYTE_CHARACTERS`], indexed by character code point.
const fn character_bytes() -> [Option<u8>; 324] {
    let mut bytes = [None; 324];
    let mut byte = 0usize;
    while byte < 256 {
        bytes[BYTE_CHARACTERS[byte] as usize] = Some(byte as u8);
        byte += 1;
    }
    bytes
}

/// Returns the byte that `character` represents in the GPT-2 byte level alphabet, if any.
fn byte_level_byte(character: char) -> Option<u8> {
    CHARACTER_BYTES.get(character as usize).copied().flatten()
}

/// Returns the fields and the `type` of the component `value` at `path` of a tokenizer definition.
fn component<'o>(value: &'o Value, path: &str) -> Result<(&'o Map<String, Value>, &'o str), TokenizerError> {
    match value {
        Value::Object(fields) => match fields.get("type") {
            Some(Value::String(r#type)) => Ok((fields, r#type)),
            _ => Err(TokenizerError::InvalidDefinition { message: format!("`{path}.type` must be a string") }),
        },
        _ => Err(TokenizerError::InvalidDefinition { message: format!("`{path}` must be an object") }),
    }
}

/// Returns the error for the component at `path`, whose type `type` is not supported.
fn unsupported_type(path: &str, r#type: &str) -> TokenizerError {
    TokenizerError::UnsupportedDefinition { message: format!("`{path}` of type `{type}` is not supported") }
}

/// Validates that the option `key` of `object` equals `expected`.
///
/// # Parameters
///
///   - `object`: Component of a tokenizer definition that holds the option.
///   - `key`: Name of the option.
///   - `expected`: Only supported value of the option.
///   - `default`: Value that the option takes when `object` does not specify it.
///   - `path`: Path of the option in the tokenizer definition, used in the error message.
fn validate_option(
    object: &Map<String, Value>,
    key: &str,
    expected: &Value,
    default: &Value,
    path: &str,
) -> Result<(), TokenizerError> {
    let value = object.get(key).unwrap_or(default);
    if value == expected {
        Ok(())
    } else {
        Err(TokenizerError::UnsupportedDefinition {
            message: format!("`{path}` must be `{expected}` but is `{value}`"),
        })
    }
}

/// Returns the token identifier that `value` holds, or [`None`] if it is not a 32-bit unsigned integer.
fn token_id_from_value(value: &Value) -> Option<TokenId> {
    value.as_u64().and_then(|id| TokenId::try_from(id).ok())
}

/// Parses the `pre_tokenizer` component of a tokenizer definition and returns the patterns of its `Split` steps.
fn parse_pre_tokenizer(pre_tokenizer: &Value) -> Result<Vec<Regex>, TokenizerError> {
    let structure_error = || TokenizerError::UnsupportedDefinition {
        message: "`pre_tokenizer` must be a `ByteLevel` pre-tokenizer or a `Sequence` of `Split` pre-tokenizers \
            followed by one `ByteLevel` pre-tokenizer"
            .to_string(),
    };
    if pre_tokenizer.is_null() {
        return Err(structure_error());
    }
    let steps = match component(pre_tokenizer, "pre_tokenizer")? {
        (fields, "Sequence") => match fields.get("pretokenizers") {
            Some(Value::Array(steps)) if !steps.is_empty() => steps
                .iter()
                .enumerate()
                .map(|(index, step)| (step, format!("pre_tokenizer.pretokenizers[{index}]")))
                .collect::<Vec<_>>(),
            Some(Value::Array(_)) => return Err(structure_error()),
            _ => {
                return Err(TokenizerError::InvalidDefinition {
                    message: "`pre_tokenizer.pretokenizers` must be an array".to_string(),
                });
            }
        },
        _ => vec![(pre_tokenizer, "pre_tokenizer".to_string())],
    };
    let mut split_patterns = Vec::new();
    for (index, (step, path)) in steps.iter().enumerate() {
        let last = index + 1 == steps.len();
        match component(step, path)? {
            (fields, "Split") if !last => {
                let pattern = match fields.get("pattern") {
                    Some(Value::Object(pattern)) => match pattern.get("Regex") {
                        Some(Value::String(pattern)) => pattern,
                        _ => {
                            return Err(TokenizerError::UnsupportedDefinition {
                                message: format!("`{path}.pattern` must be a `Regex` pattern"),
                            });
                        }
                    },
                    _ => {
                        return Err(TokenizerError::InvalidDefinition {
                            message: format!("`{path}.pattern` must be an object"),
                        });
                    }
                };
                let isolated = Value::String("Isolated".to_string());
                validate_option(fields, "behavior", &isolated, &Value::Null, &format!("{path}.behavior"))?;
                validate_option(fields, "invert", &Value::Bool(false), &Value::Bool(false), &format!("{path}.invert"))?;
                split_patterns.push(Regex::new(pattern).map_err(|error| TokenizerError::InvalidDefinition {
                    message: format!("`{path}.pattern` is not a valid regular expression: {error}"),
                })?);
            }
            (fields, "ByteLevel") if last => {
                // The reference implementation enables both options by default.
                for key in ["add_prefix_space", "use_regex"] {
                    validate_option(fields, key, &Value::Bool(false), &Value::Bool(true), &format!("{path}.{key}"))?;
                }
            }
            (_, "Split" | "ByteLevel") => return Err(structure_error()),
            (_, r#type) => return Err(unsupported_type(path, r#type)),
        }
    }
    Ok(split_patterns)
}

/// Parses the `model.vocab` field of a tokenizer definition and returns the identifiers of the model tokens keyed by
/// their strings, together with the vocabulary entries indexed by identifier.
fn parse_vocabulary(
    vocabulary: Option<Value>,
) -> Result<(HashMap<String, TokenId>, Vec<Option<VocabularyEntry>>), TokenizerError> {
    let Some(Value::Object(vocabulary)) = vocabulary else {
        return Err(TokenizerError::InvalidDefinition { message: "`model.vocab` must be an object".to_string() });
    };
    let mut ids = HashMap::with_capacity(vocabulary.len());
    let mut entries = Vec::<Option<VocabularyEntry>>::with_capacity(vocabulary.len());
    for (token, id) in vocabulary {
        let id = token_id_from_value(&id).ok_or_else(|| TokenizerError::InvalidDefinition {
            message: format!("`model.vocab` maps `{token}` to `{id}`, which is not a 32-bit unsigned integer"),
        })?;
        let index = id as usize;
        if entries.len() <= index {
            entries.resize_with(index + 1, || None);
        }
        if let Some(entry) = &entries[index] {
            return Err(TokenizerError::InvalidDefinition {
                message: format!("`model.vocab` maps both `{}` and `{token}` to {id}", entry.content),
            });
        }
        entries[index] = Some(VocabularyEntry { content: token.clone(), added: false, special: false });
        ids.insert(token, id);
    }
    Ok((ids, entries))
}

/// Parses the `model.merges` field of a tokenizer definition and returns the merges keyed by the pair of token
/// identifiers that they merge.
fn parse_merges(
    merges: Option<&Value>,
    vocabulary: &HashMap<String, TokenId>,
) -> Result<HashMap<(TokenId, TokenId), Merge>, TokenizerError> {
    let Some(Value::Array(merges)) = merges else {
        return Err(TokenizerError::InvalidDefinition { message: "`model.merges` must be an array".to_string() });
    };
    let mut parsed_merges = HashMap::with_capacity(merges.len());
    for (rank, merge) in merges.iter().enumerate() {
        let pair = match merge {
            Value::Array(pair) => match pair.as_slice() {
                [Value::String(left), Value::String(right)] => Some((left.as_str(), right.as_str())),
                _ => None,
            },
            Value::String(pair) => pair.split_once(' ').filter(|(_, right)| !right.contains(' ')),
            _ => None,
        };
        let Some((left, right)) = pair else {
            return Err(TokenizerError::InvalidDefinition {
                message: format!("`model.merges[{rank}]` must be a pair of tokens"),
            });
        };
        let token_id = |token: &str| {
            vocabulary.get(token).copied().ok_or_else(|| TokenizerError::InvalidDefinition {
                message: format!("`model.merges[{rank}]` refers to `{token}`, which is not in `model.vocab`"),
            })
        };
        let pair = (token_id(left)?, token_id(right)?);
        let id = token_id(&format!("{left}{right}"))?;
        // Like the reference implementation, a repeated merge keeps the rank of its last occurrence.
        parsed_merges.insert(pair, Merge { rank, id });
    }
    Ok(parsed_merges)
}

/// Returns a pattern that matches any of `contents` at the leftmost possible position, preferring the longest content
/// among those that match at that position, or [`None`] if `contents` is empty.
fn added_token_pattern(mut contents: Vec<String>) -> Result<Option<Regex>, TokenizerError> {
    if contents.is_empty() {
        return Ok(None);
    }
    // Alternations match their first matching alternative, so listing longer contents first prefers them.
    contents.sort_by_key(|content| Reverse(content.len()));
    let pattern = contents.iter().map(|content| fancy_regex::escape(content)).collect::<Vec<_>>().join("|");
    Regex::new(&pattern).map(Some).map_err(|error| TokenizerError::InvalidDefinition {
        message: format!("the contents of `added_tokens` do not form a valid pattern: {error}"),
    })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    /// Text with the tokens that the reference `tokenizers` library produces for it with the tokenizer of the Inkling 2
    /// Mini release.
    struct ReleaseParityCase {
        /// Text to encode.
        text: &'static str,

        /// Tokens of the text when special tokens are parsed.
        tokens: &'static [TokenId],

        /// Tokens of the text when special tokens are treated as ordinary text.
        literal_tokens: &'static [TokenId],

        /// Text that the tokens decode to.
        decoded: &'static str,
    }

    /// Cases of the release parity test, which cover empty text, whitespace, digits, contractions, scripts without word
    /// boundaries, emoji, and special tokens.
    const RELEASE_PARITY_CASES: &[ReleaseParityCase] = &[
        ReleaseParityCase { text: "", tokens: &[], literal_tokens: &[], decoded: "" },
        ReleaseParityCase {
            text: "Hello, world!",
            tokens: &[32269, 11, 150770, 0],
            literal_tokens: &[32269, 11, 150770, 0],
            decoded: "Hello, world!",
        },
        ReleaseParityCase {
            text: " leading and trailing spaces  ",
            tokens: &[154384, 149880, 162610, 279, 156465, 149859],
            literal_tokens: &[154384, 149880, 162610, 279, 156465, 149859],
            decoded: " leading and trailing spaces  ",
        },
        ReleaseParityCase {
            text: "The quick brown fox jumps over the lazy dog's back; it's 12345 o'clock.",
            tokens: &[
                371, 157021, 161319, 220, 32329, 160581, 82, 150053, 149876, 161703, 153350, 6, 82, 150387, 26, 149921,
                6, 82, 220, 70904, 68358, 149896, 6, 4435, 13,
            ],
            literal_tokens: &[
                371, 157021, 161319, 220, 32329, 160581, 82, 150053, 149876, 161703, 153350, 6, 82, 150387, 26, 149921,
                6, 82, 220, 70904, 68358, 149896, 6, 4435, 13,
            ],
            decoded: "The quick brown fox jumps over the lazy dog's back; it's 12345 o'clock.",
        },
        ReleaseParityCase {
            text: "def f(x):\n    return x ** 2  # squares\n\n\nprint(f(3.14159))\n",
            tokens: &[
                1719, 149983, 7, 87, 8, 25, 198, 149862, 150583, 149924, 197033, 220, 17, 220, 196994, 165839, 68019,
                1160, 7, 69, 7, 18, 13, 73388, 69022, 8, 8, 198,
            ],
            literal_tokens: &[
                1719, 149983, 7, 87, 8, 25, 198, 149862, 150583, 149924, 197033, 220, 17, 220, 196994, 165839, 68019,
                1160, 7, 69, 7, 18, 13, 73388, 69022, 8, 8, 198,
            ],
            decoded: "def f(x):\n    return x ** 2  # squares\n\n\nprint(f(3.14159))\n",
        },
        ReleaseParityCase {
            text: "Tabs\tand\r\nCRLF line endings\r\n\r\n",
            tokens: &[51, 31029, 197, 276, 68247, 145969, 146283, 150264, 150255, 68607, 70392],
            literal_tokens: &[51, 31029, 197, 276, 68247, 145969, 146283, 150264, 150255, 68607, 70392],
            decoded: "Tabs\tand\r\nCRLF line endings\r\n\r\n",
        },
        ReleaseParityCase {
            text: "na\u{ef}ve caf\u{e9} d\u{e9}j\u{e0} vu \u{2014} \u{201c}quoted\u{201d} \u{2026} ellipsis",
            tokens: &[
                1364, 5169, 290, 162389, 154919, 152684, 150220, 150060, 32112, 3791, 23068, 151105, 157916, 107201,
            ],
            literal_tokens: &[
                1364, 5169, 290, 162389, 154919, 152684, 150220, 150060, 32112, 3791, 23068, 151105, 157916, 107201,
            ],
            decoded: "na\u{ef}ve caf\u{e9} d\u{e9}j\u{e0} vu \u{2014} \u{201c}quoted\u{201d} \u{2026} ellipsis",
        },
        ReleaseParityCase {
            text: concat!(
                "\u{65e5}\u{672c}\u{8a9e}\u{306e}\u{30c6}\u{30ad}\u{30b9}\u{30c8}\u{3068}\u{4e2d}\u{6587}\u{6587}",
                "\u{672c}\u{3001}\u{d55c}\u{ad6d}\u{c5b4} \u{d14d}\u{c2a4}\u{d2b8}",
            ),
            tokens: &[
                90626, 22912, 6084, 2804, 2064, 3527, 3529, 6080, 114903, 18004, 18035, 6433, 91422, 27040, 220, 27239,
                86274,
            ],
            literal_tokens: &[
                90626, 22912, 6084, 2804, 2064, 3527, 3529, 6080, 114903, 18004, 18035, 6433, 91422, 27040, 220, 27239,
                86274,
            ],
            decoded: concat!(
                "\u{65e5}\u{672c}\u{8a9e}\u{306e}\u{30c6}\u{30ad}\u{30b9}\u{30c8}\u{3068}\u{4e2d}\u{6587}\u{6587}",
                "\u{672c}\u{3001}\u{d55c}\u{ad6d}\u{c5b4} \u{d14d}\u{c2a4}\u{d2b8}",
            ),
        },
        ReleaseParityCase {
            text: "emoji \u{1f642}\u{1f680}\u{1f469}\u{200d}\u{1f4bb} and symbols \u{b1}\u{d7}\u{f7}\u{2248}\u{221e}",
            tokens: &[
                40688, 220, 104523, 108185, 74269, 70363, 111037, 149880, 165269, 154811, 5209, 5177, 26276, 25183,
            ],
            literal_tokens: &[
                40688, 220, 104523, 108185, 74269, 70363, 111037, 149880, 165269, 154811, 5209, 5177, 26276, 25183,
            ],
            decoded: concat!(
                "emoji \u{1f642}\u{1f680}\u{1f469}\u{200d}\u{1f4bb} and symbols ",
                "\u{b1}\u{d7}\u{f7}\u{2248}\u{221e}",
            ),
        },
        ReleaseParityCase {
            text: "MixedCASEWords HTTPServer camelCaseIdentifier snake_case_identifier",
            tokens: &[33110, 302, 147125, 103634, 198447, 52861, 220, 39907, 4474, 88051, 189771, 62, 1413, 62, 86530],
            literal_tokens: &[
                33110, 302, 147125, 103634, 198447, 52861, 220, 39907, 4474, 88051, 189771, 62, 1413, 62, 86530,
            ],
            decoded: "MixedCASEWords HTTPServer camelCaseIdentifier snake_case_identifier",
        },
        ReleaseParityCase {
            text: "1234567890 3.14 1e-5 0xDEADBEEF 1,000,000",
            tokens: &[
                70904, 81554, 91927, 15, 220, 18, 13, 68084, 220, 16, 68, 12, 20, 220, 15, 87, 145890, 145903, 146053,
                146221, 220, 16, 11, 68128, 11, 68128,
            ],
            literal_tokens: &[
                70904, 81554, 91927, 15, 220, 18, 13, 68084, 220, 16, 68, 12, 20, 220, 15, 87, 145890, 145903, 146053,
                146221, 220, 16, 11, 68128, 11, 68128,
            ],
            decoded: "1234567890 3.14 1e-5 0xDEADBEEF 1,000,000",
        },
        ReleaseParityCase {
            text: "I'm you're we've they'll it'd DON'T",
            tokens: &[40, 6, 76, 149941, 6, 261, 149961, 6, 290, 150051, 6, 359, 149921, 6, 67, 199445, 6, 51],
            literal_tokens: &[40, 6, 76, 149941, 6, 261, 149961, 6, 290, 150051, 6, 359, 149921, 6, 67, 199445, 6, 51],
            decoded: "I'm you're we've they'll it'd DON'T",
        },
        ReleaseParityCase {
            text: "    \n\n   \t  \n",
            tokens: &[149860, 68014, 149862, 197, 149865],
            literal_tokens: &[149860, 68014, 149862, 197, 149865],
            decoded: "    \n\n   \t  \n",
        },
        ReleaseParityCase {
            text: "https://example.com/path?query=1&other=two#fragment",
            tokens: &[68713, 25, 145055, 1805, 13, 376, 14, 1448, 30, 35790, 28, 16, 5, 995, 28, 1286, 2, 34694],
            literal_tokens: &[
                68713, 25, 145055, 1805, 13, 376, 14, 1448, 30, 35790, 28, 16, 5, 995, 28, 1286, 2, 34694,
            ],
            decoded: "https://example.com/path?query=1&other=two#fragment",
        },
        ReleaseParityCase {
            text: "<|end_message|> looks like a special token <|unused_200054|>",
            tokens: &[200010, 155974, 150183, 149878, 151035, 152996, 220, 200054],
            literal_tokens: &[
                27, 91, 406, 62, 65752, 91, 29, 155974, 150183, 149878, 151035, 152996, 196982, 91, 130041, 62, 68058,
                99363, 91, 29,
            ],
            decoded: "<|end_message|> looks like a special token <|unused_200054|>",
        },
        ReleaseParityCase {
            text: "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"days\":3}}",
            tokens: &[
                90, 1, 930, 1, 25, 1, 832, 62, 30847, 1, 11, 1, 70747, 1, 25, 90, 1, 2958, 1, 25, 1, 3967, 1, 11, 1,
                70095, 1, 25, 18, 92, 92,
            ],
            literal_tokens: &[
                90, 1, 930, 1, 25, 1, 832, 62, 30847, 1, 11, 1, 70747, 1, 25, 90, 1, 2958, 1, 25, 1, 3967, 1, 11, 1,
                70095, 1, 25, 18, 92, 92,
            ],
            decoded: "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"days\":3}}",
        },
        ReleaseParityCase {
            text: concat!(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ababababababababababababababa",
                "bababababababababababababababababababababababababab",
            ),
            tokens: &[
                3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642,
                3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642,
                3642, 150189, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332,
                332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332,
                332,
            ],
            literal_tokens: &[
                3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642,
                3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642, 3642,
                3642, 150189, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332,
                332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332,
                332,
            ],
            decoded: concat!(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ababababababababababababababa",
                "bababababababababababababababababababababababababab",
            ),
        },
        ReleaseParityCase {
            text: concat!(
                "\u{dc}n\u{ef}c\u{f6}d\u{e9} \u{160}\u{164}\u{158}\u{cd}\u{147}\u{11e} \u{395}\u{3bb}\u{3bb}\u{3b7}",
                "\u{3bd}\u{3b9}\u{3ba}\u{3ac} \u{43a}\u{438}\u{440}\u{438}\u{43b}\u{43b}\u{438}\u{446}\u{430} \u{5e2}",
                "\u{5d1}\u{5e8}\u{5d9}\u{5ea} \u{627}\u{644}\u{639}\u{631}\u{628}\u{64a}\u{629}",
            ),
            tokens: &[
                5214, 77, 138472, 5176, 68096, 152229, 5287, 5337, 5199, 5321, 5280, 151001, 110549, 5367, 82530,
                156695, 118272, 111450, 152194, 117966, 168984,
            ],
            literal_tokens: &[
                5214, 77, 138472, 5176, 68096, 152229, 5287, 5337, 5199, 5321, 5280, 151001, 110549, 5367, 82530,
                156695, 118272, 111450, 152194, 117966, 168984,
            ],
            decoded: concat!(
                "\u{dc}n\u{ef}c\u{f6}d\u{e9} \u{160}\u{164}\u{158}\u{cd}\u{147}\u{11e} \u{395}\u{3bb}\u{3bb}\u{3b7}",
                "\u{3bd}\u{3b9}\u{3ba}\u{3ac} \u{43a}\u{438}\u{440}\u{438}\u{43b}\u{43b}\u{438}\u{446}\u{430} \u{5e2}",
                "\u{5d1}\u{5e8}\u{5d9}\u{5ea} \u{627}\u{644}\u{639}\u{631}\u{628}\u{64a}\u{629}",
            ),
        },
    ];

    /// Returns a small synthetic tokenizer definition with byte level BPE, three merges, and three added tokens.
    fn definition() -> Value {
        json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 15,
                    "content": "<c>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true,
                },
                {
                    "id": 16,
                    "content": "<c>!",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true,
                },
                {
                    "id": 17,
                    "content": "<b>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": true,
                    "special": false,
                },
            ],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": { "Regex": " ?\\p{L}+|\\s+(?!\\S)|\\s+|[^\\s\\p{L}]+" },
                        "behavior": "Isolated",
                        "invert": false,
                    },
                    { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false },
                ],
            },
            "post_processor": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": false,
                "use_regex": true,
            },
            "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": true,
                "vocab": {
                    "a": 0,
                    "b": 1,
                    "c": 2,
                    "Ġ": 3,
                    "!": 4,
                    "<": 5,
                    ">": 6,
                    "ab": 7,
                    "abc": 8,
                    "Ġab": 9,
                    "Ã": 10,
                    "©": 11,
                    "Ã©": 12,
                    "ca": 13,
                    "€": 14,
                },
                "merges": [["a", "b"], ["Ġ", "ab"], ["ab", "c"], "Ã ©"],
            },
        })
    }

    /// Returns the [`HuggingFaceTokenizer`] of [`definition`] after applying `modify` to that definition.
    fn modified_tokenizer(modify: impl FnOnce(&mut Value)) -> Result<HuggingFaceTokenizer, TokenizerError> {
        let mut definition = definition();
        modify(&mut definition);
        HuggingFaceTokenizer::from_json(&definition.to_string())
    }

    #[test]
    fn test_hugging_face_tokenizer_from_json() {
        let tokenizer = HuggingFaceTokenizer::from_json(&definition().to_string()).unwrap();
        assert_eq!(tokenizer.vocabulary_size(), 18);
        assert_eq!(tokenizer.encode("abc", SpecialTokenHandling::Parse), Ok(vec![8]));

        // Missing optional components and options take their defaults.
        let tokenizer = modified_tokenizer(|definition| {
            let definition = definition.as_object_mut().unwrap();
            for key in ["truncation", "padding", "normalizer", "post_processor", "added_tokens"] {
                definition.remove(key);
            }
            let model = definition["model"].as_object_mut().unwrap();
            for key in ["dropout", "unk_token", "continuing_subword_prefix", "fuse_unk", "byte_fallback"] {
                model.remove(key);
            }
        })
        .unwrap();
        assert_eq!(tokenizer.vocabulary_size(), 15);
        assert_eq!(tokenizer.encode("<c>", SpecialTokenHandling::Parse), Ok(vec![5, 2, 6]));
    }

    #[test]
    fn test_hugging_face_tokenizer_from_json_unsupported_definitions() {
        assert!(matches!(
            modified_tokenizer(|definition| definition["normalizer"] = json!({ "type": "Lowercase" })),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`normalizer` must be `null` but is `{\"type\":\"Lowercase\"}`",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["truncation"] = json!({ "max_length": 8 })),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`truncation` must be `null` but is `{\"max_length\":8}`",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["pre_tokenizer"] = Value::Null),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`pre_tokenizer` must be a `ByteLevel` pre-tokenizer or a `Sequence` of `Split` \
                    pre-tokenizers followed by one `ByteLevel` pre-tokenizer",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["pre_tokenizer"]["pretokenizers"][1]["type"] = json!("Split")),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`pre_tokenizer` must be a `ByteLevel` pre-tokenizer or a `Sequence` of `Split` \
                    pre-tokenizers followed by one `ByteLevel` pre-tokenizer",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["pre_tokenizer"] = json!({ "type": "Whitespace" })),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`pre_tokenizer` of type `Whitespace` is not supported",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| {
                definition["pre_tokenizer"]["pretokenizers"][0] = json!({ "type": "Digits" });
            }),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`pre_tokenizer.pretokenizers[0]` of type `Digits` is not supported",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| {
                definition["pre_tokenizer"]["pretokenizers"][0]["pattern"] = json!({ "String": " " });
            }),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`pre_tokenizer.pretokenizers[0].pattern` must be a `Regex` pattern",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| {
                definition["pre_tokenizer"]["pretokenizers"][0]["behavior"] = json!("Removed");
            }),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`pre_tokenizer.pretokenizers[0].behavior` must be `\"Isolated\"` but is `\"Removed\"`",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["pre_tokenizer"]["pretokenizers"][0]["invert"] = json!(true)),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`pre_tokenizer.pretokenizers[0].invert` must be `false` but is `true`",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| {
                definition["pre_tokenizer"]["pretokenizers"][1].as_object_mut().unwrap().remove("use_regex");
            }),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`pre_tokenizer.pretokenizers[1].use_regex` must be `false` but is `true`",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["post_processor"] = json!({ "type": "TemplateProcessing" })),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`post_processor` of type `TemplateProcessing` is not supported",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["decoder"] = Value::Null),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`decoder` must be a `ByteLevel` decoder",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["decoder"] = json!({ "type": "WordPiece" })),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`decoder` of type `WordPiece` is not supported",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["model"]["type"] = json!("Unigram")),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`model` of type `Unigram` is not supported",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["model"]["dropout"] = json!(0.1)),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`model.dropout` must be `null` but is `0.1`",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["model"]["continuing_subword_prefix"] = json!("##")),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`model.continuing_subword_prefix` must be `null` but is `\"##\"`",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["model"]["byte_fallback"] = json!(true)),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`model.byte_fallback` must be `false` but is `true`",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["added_tokens"][0]["lstrip"] = json!(true)),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`added_tokens[0].lstrip` must be `false` but is `true`",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["added_tokens"][0]["normalized"] = json!(true)),
            Err(TokenizerError::UnsupportedDefinition { message })
                if message == "`added_tokens[0].normalized` must be `false` but is `true`",
        ));
    }

    #[test]
    fn test_hugging_face_tokenizer_from_json_invalid_definitions() {
        assert!(matches!(
            HuggingFaceTokenizer::from_json("{"),
            Err(TokenizerError::InvalidDefinition { message })
                if message == "EOF while parsing an object at line 1 column 1",
        ));
        assert!(matches!(
            HuggingFaceTokenizer::from_json("[]"),
            Err(TokenizerError::InvalidDefinition { message })
                if message == "the tokenizer definition must be a JSON object",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["decoder"] = json!({})),
            Err(TokenizerError::InvalidDefinition { message }) if message == "`decoder.type` must be a string",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["model"] = json!([])),
            Err(TokenizerError::InvalidDefinition { message }) if message == "`model` must be an object",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| {
                definition["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = json!("(");
            }),
            Err(TokenizerError::InvalidDefinition { message })
                if message == "`pre_tokenizer.pretokenizers[0].pattern` is not a valid regular expression: Parsing \
                    error at position 1: Opening parenthesis without closing parenthesis",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["model"]["ignore_merges"] = json!("yes")),
            Err(TokenizerError::InvalidDefinition { message }) if message == "`model.ignore_merges` must be a boolean",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["model"]["vocab"]["d"] = json!(-1)),
            Err(TokenizerError::InvalidDefinition { message })
                if message == "`model.vocab` maps `d` to `-1`, which is not a 32-bit unsigned integer",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["model"]["vocab"]["d"] = json!(0)),
            Err(TokenizerError::InvalidDefinition { message }) if message == "`model.vocab` maps both `a` and `d` to 0",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["model"]["merges"] = json!({})),
            Err(TokenizerError::InvalidDefinition { message }) if message == "`model.merges` must be an array",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["model"]["merges"][3] = json!("Ã © ©")),
            Err(TokenizerError::InvalidDefinition { message })
                if message == "`model.merges[3]` must be a pair of tokens",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["model"]["merges"][3] = json!(["c", "a", "b"])),
            Err(TokenizerError::InvalidDefinition { message })
                if message == "`model.merges[3]` must be a pair of tokens",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["model"]["merges"][3] = json!(["d", "a"])),
            Err(TokenizerError::InvalidDefinition { message })
                if message == "`model.merges[3]` refers to `d`, which is not in `model.vocab`",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["model"]["merges"][3] = json!(["b", "a"])),
            Err(TokenizerError::InvalidDefinition { message })
                if message == "`model.merges[3]` refers to `ba`, which is not in `model.vocab`",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["added_tokens"] = json!({})),
            Err(TokenizerError::InvalidDefinition { message }) if message == "`added_tokens` must be an array",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["added_tokens"][1]["id"] = json!("16")),
            Err(TokenizerError::InvalidDefinition { message })
                if message == "`added_tokens[1].id` must be a 32-bit unsigned integer",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["added_tokens"][1]["content"] = json!("")),
            Err(TokenizerError::InvalidDefinition { message })
                if message == "`added_tokens[1].content` must be a non-empty string",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["added_tokens"][1]["special"] = json!(1)),
            Err(TokenizerError::InvalidDefinition { message })
                if message == "`added_tokens[1].special` must be a boolean",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["added_tokens"][1]["id"] = json!(15)),
            Err(TokenizerError::InvalidDefinition { message })
                if message == "`added_tokens[1]` reuses the identifier 15 of added token `<c>`",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["added_tokens"][1]["id"] = json!(3)),
            Err(TokenizerError::InvalidDefinition { message })
                if message == "`added_tokens[1]` reuses the identifier 3 of model token `Ġ`",
        ));
        assert!(matches!(
            modified_tokenizer(|definition| definition["added_tokens"][1]["content"] = json!("<c>")),
            Err(TokenizerError::InvalidDefinition { message })
                if message == "`added_tokens[1]` repeats the content `<c>` of an earlier added token",
        ));

        // An added token may reuse the identifier of the model token with the same content.
        let tokenizer = modified_tokenizer(|definition| {
            definition["added_tokens"][1]["id"] = json!(7);
            definition["added_tokens"][1]["content"] = json!("ab");
        })
        .unwrap();
        assert_eq!(tokenizer.encode("abc", SpecialTokenHandling::Parse), Ok(vec![7, 2]));
        assert_eq!(tokenizer.decode(&[7, 2], true), Ok("c".to_string()));
    }

    #[test]
    fn test_hugging_face_tokenizer_from_file() {
        let directory = std::env::temp_dir().join(format!("ryft_models_tokenizer_{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("tokenizer.json");
        std::fs::write(&path, definition().to_string()).unwrap();
        let tokenizer = HuggingFaceTokenizer::from_file(&path).unwrap();
        assert_eq!(tokenizer.encode(" ab", SpecialTokenHandling::Parse), Ok(vec![9]));
        std::fs::remove_dir_all(&directory).unwrap();

        let error = std::fs::read_to_string(&path).unwrap_err();
        assert!(matches!(
            HuggingFaceTokenizer::from_file(&path),
            Err(TokenizerError::Io { message }) if message == format!("`{}`: {error}", path.display()),
        ));
    }

    #[test]
    fn test_hugging_face_tokenizer_encode() {
        let tokenizer = HuggingFaceTokenizer::from_json(&definition().to_string()).unwrap();
        assert_eq!(tokenizer.encode("", SpecialTokenHandling::Parse), Ok(vec![]));

        // Added tokens split the text before pretokenization, preferring the longest content at a position.
        assert_eq!(tokenizer.encode("abc<c>! ab<b>é", SpecialTokenHandling::Parse), Ok(vec![8, 16, 9, 17, 12]));
        assert_eq!(tokenizer.encode("<c><c>", SpecialTokenHandling::Parse), Ok(vec![15, 15]));

        // Pieces in the vocabulary map directly to their tokens, and the others are merged by rank.
        assert_eq!(tokenizer.encode("ca", SpecialTokenHandling::Parse), Ok(vec![13]));
        assert_eq!(tokenizer.encode(" abc", SpecialTokenHandling::Parse), Ok(vec![9, 2]));
        assert_eq!(tokenizer.encode("ab  !", SpecialTokenHandling::Parse), Ok(vec![7, 3, 3, 4]));
    }

    #[test]
    fn test_hugging_face_tokenizer_encode_literal_special_tokens() {
        let tokenizer = HuggingFaceTokenizer::from_json(&definition().to_string()).unwrap();

        // Special token contents are tokenized as text, while added tokens that are not special are still matched.
        assert_eq!(
            tokenizer.encode("abc<c>! ab<b>é", SpecialTokenHandling::Literal),
            Ok(vec![8, 5, 2, 6, 4, 9, 17, 12]),
        );
    }

    #[test]
    fn test_hugging_face_tokenizer_encode_without_ignoring_merges() {
        let tokenizer = modified_tokenizer(|definition| definition["model"]["ignore_merges"] = json!(false)).unwrap();
        assert_eq!(tokenizer.encode("ca abc", SpecialTokenHandling::Parse), Ok(vec![2, 0, 9, 2]));
    }

    #[test]
    fn test_hugging_face_tokenizer_encode_byte_level_pre_tokenizer() {
        let tokenizer = modified_tokenizer(|definition| {
            definition["pre_tokenizer"] = definition["pre_tokenizer"]["pretokenizers"][1].clone();
        })
        .unwrap();
        assert_eq!(tokenizer.encode("ab ab", SpecialTokenHandling::Parse), Ok(vec![7, 9]));
    }

    #[test]
    fn test_hugging_face_tokenizer_encode_errors() {
        let tokenizer = HuggingFaceTokenizer::from_json(&definition().to_string()).unwrap();
        assert!(matches!(
            tokenizer.encode("abx", SpecialTokenHandling::Parse),
            Err(TokenizerError::Encoding { message }) if message == "the vocabulary has no token for the byte `0x78`",
        ));

        // Patterns that backtrack excessively fail instead of running for an unbounded amount of time.
        let tokenizer = modified_tokenizer(|definition| {
            definition["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = json!("(?:a(?=a)|a)*c");
        })
        .unwrap();
        assert!(matches!(
            tokenizer.encode(&"a".repeat(64), SpecialTokenHandling::Parse),
            Err(TokenizerError::Encoding { message })
                if message == "the `Split` pre-tokenizer pattern failed: Error executing regex: Max limit for \
                    backtracking count exceeded",
        ));
    }

    #[test]
    fn test_hugging_face_tokenizer_decode() {
        let tokenizer = HuggingFaceTokenizer::from_json(&definition().to_string()).unwrap();
        assert_eq!(tokenizer.decode(&[], false), Ok(String::new()));
        assert_eq!(tokenizer.decode(&[8, 16, 9, 17, 12], false), Ok("abc<c>! ab<b>é".to_string()));
        assert_eq!(tokenizer.decode(&[8, 16, 9, 17, 12], true), Ok("abc ab<b>é".to_string()));

        // Multibyte characters may span tokens, and invalid UTF-8 becomes replacement characters.
        assert_eq!(tokenizer.decode(&[10, 11], false), Ok("é".to_string()));
        assert_eq!(tokenizer.decode(&[10, 15, 11], false), Ok("\u{fffd}<c>\u{fffd}".to_string()));

        // Tokens with characters outside the byte level alphabet contribute their UTF-8 bytes.
        assert_eq!(tokenizer.decode(&[14], false), Ok("€".to_string()));

        assert!(matches!(
            tokenizer.decode(&[0, 18], false),
            Err(TokenizerError::Decoding { message }) if message == "unknown token identifier 18",
        ));
    }

    #[test]
    fn test_hugging_face_tokenizer_vocabulary_size() {
        let tokenizer = HuggingFaceTokenizer::from_json(&definition().to_string()).unwrap();
        assert_eq!(tokenizer.vocabulary_size(), 18);
    }

    #[test]
    fn test_hugging_face_tokenizer_token_id() {
        let tokenizer = HuggingFaceTokenizer::from_json(&definition().to_string()).unwrap();
        assert_eq!(tokenizer.token_id("Ġab"), Some(9));
        assert_eq!(tokenizer.token_id("<c>"), Some(15));
        assert_eq!(tokenizer.token_id("<b>"), Some(17));
        assert_eq!(tokenizer.token_id(" ab"), None);
    }

    #[test]
    fn test_hugging_face_tokenizer_token() {
        let tokenizer = HuggingFaceTokenizer::from_json(&definition().to_string()).unwrap();
        assert_eq!(tokenizer.token(9), Some("Ġab"));
        assert_eq!(tokenizer.token(16), Some("<c>!"));
        assert_eq!(tokenizer.token(18), None);
    }

    #[test]
    #[ignore = "requires the Inkling 2 release directory in `RYFT_MODELS_INKLING2_RELEASE_DIRECTORY`"]
    fn test_hugging_face_tokenizer_release_parity() {
        let directory = std::env::var("RYFT_MODELS_INKLING2_RELEASE_DIRECTORY").unwrap();
        let tokenizer = HuggingFaceTokenizer::from_file(Path::new(&directory).join("tokenizer.json")).unwrap();
        assert_eq!(tokenizer.vocabulary_size(), 200_058);
        for case in RELEASE_PARITY_CASES {
            let text = case.text;
            assert_eq!(tokenizer.encode(text, SpecialTokenHandling::Parse), Ok(case.tokens.to_vec()), "{text:?}");
            assert_eq!(
                tokenizer.encode(text, SpecialTokenHandling::Literal),
                Ok(case.literal_tokens.to_vec()),
                "{text:?}",
            );
            assert_eq!(tokenizer.decode(case.tokens, false), Ok(case.decoded.to_string()), "{text:?}");
        }
    }
}
