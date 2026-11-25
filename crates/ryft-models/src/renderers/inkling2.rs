//! [`Inkling2Renderer`], which implements [`Renderer`] for the conversation format of Inkling 2 models.

use std::ops::Range;

use crate::json::JsonValue;
use crate::renderers::{
    ContentPart, Conversation, Message, ReasoningEffort, RenderedConversation, Renderer, RendererError, Role, ToolCall,
};
use crate::tokenizers::{SpecialTokenHandling, TokenId, Tokenizer};

// TODO(eaplatanios): Review this module.

/// [`Renderer`] for the conversation format of Inkling 2 models, which matches the chat template of the Inkling 2
/// release.
///
/// # Format
///
/// A rendered conversation is a sequence of blocks. Each block consists of a role token (e.g., `<|message_user|>`),
/// optional author name text, a content token (e.g., `<|content_text|>`), payload text, and `<|end_message|>`. Author
/// names and payloads are encoded separately with [`SpecialTokenHandling::Literal`], so text that spells a special
/// token cannot inject that token. A conversation is rendered as follows.
///
///   - If the conversation declares tools, it starts with the block `<|message_system|>tool_declare<|content_xml|>`
///     whose payload is the JSON array of the tool specifications (i.e., objects with the keys `description`, `name`,
///     `parameters`, and `type`), serialized compactly with all object keys sorted.
///   - The block `<|message_system|><|content_text|>Thinking effort level: {effort}<|end_message|>` immediately
///     precedes the first message that is not a system message, or ends the conversation if there is no such message.
///     The effort is `0.9` by default, and the named efforts `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and
///     `max` are `0`, `0.1`, `0.2`, `0.7`, `0.9`, `0.99`, and `0.99`, respectively. Numeric efforts must be in
///     `[0.0, 0.99]` and are spelled like Python floats, except that zero is spelled `0`.
///   - The header of the blocks of a message is its role token followed by an author name for tool messages and named
///     assistant messages. The author of a tool message is its name or, if it has none, the function names of the
///     assistant tool calls in the conversation whose identifiers equal its tool call identifier. Other metadata
///     that the format has no place for (e.g., the names of system and user messages) is not rendered.
///   - The reasoning of an assistant message, if it is not empty, is a `<|content_thinking|>` block.
///   - Each text part is a `<|content_text|>` block, and each image part is a `<|content_image|>` block whose payload
///     consists of `token_count` `<|unused_200054|>` placeholder tokens. Images are only allowed in user and tool
///     messages. An assistant message whose content is one empty text part has no content block when it has tool
///     calls or reasoning.
///   - Each tool call of an assistant message is a block with the header `<|message_model|>` followed by the function
///     name, the content token `<|content_invoke_tool_json|>`, and the payload `{"name":...,"args":{...}}`. The keys
///     of the arguments are sorted, and the values of the arguments are serialized compactly with their own keys in
///     insertion order.
///   - Every assistant message ends with `<|content_model_end_sampling|>`, which is also the stop token.
///
/// The supervision mask is `true` for exactly the tokens of assistant messages, from their first block through
/// `<|content_model_end_sampling|>`. JSON is spelled like Python's `json.dumps` with `ensure_ascii=False`, as described
/// in [`JsonValue::to_json`].
///
/// # Parsing
///
/// [`Renderer::parse`] reads blocks of the form `[<|message_model|> [author name]] content_token payload
/// <|end_message|>` until the end of the tokens or the stop token. The model emits the leading `<|message_model|>`,
/// but the parser also accepts blocks without it. `<|content_thinking|>` blocks form the reasoning of the message,
/// `<|content_text|>`, `<|content_tool_error|>`, and `<|content_invoke_tool_text|>` blocks become text parts, and
/// `<|content_invoke_tool_json|>` blocks become tool calls. Author names are ignored, like the response template of the
/// Inkling 2 release does.
#[derive(Clone, Debug)]
pub struct Inkling2Renderer<T: Tokenizer> {
    /// Tokenizer of the model.
    tokenizer: T,

    /// Identifiers of the special tokens of the format.
    special_tokens: Inkling2SpecialTokens,
}

impl<T: Tokenizer> Inkling2Renderer<T> {
    /// Creates an [`Inkling2Renderer`] that uses `tokenizer`, which must have all the special tokens of the format.
    pub fn new(tokenizer: T) -> Result<Self, RendererError> {
        let token_id = |token: &str| {
            tokenizer
                .token_id(token)
                .ok_or_else(|| RendererError::MissingSpecialToken { token: token.to_string() })
        };
        let special_tokens = Inkling2SpecialTokens {
            message_user: token_id("<|message_user|>")?,
            message_model: token_id("<|message_model|>")?,
            message_system: token_id("<|message_system|>")?,
            message_tool: token_id("<|message_tool|>")?,
            content_text: token_id("<|content_text|>")?,
            content_image: token_id("<|content_image|>")?,
            content_thinking: token_id("<|content_thinking|>")?,
            content_xml: token_id("<|content_xml|>")?,
            content_invoke_tool_json: token_id("<|content_invoke_tool_json|>")?,
            content_invoke_tool_text: token_id("<|content_invoke_tool_text|>")?,
            content_tool_error: token_id("<|content_tool_error|>")?,
            end_message: token_id("<|end_message|>")?,
            content_model_end_sampling: token_id("<|content_model_end_sampling|>")?,
            image_placeholder: token_id("<|unused_200054|>")?,
        };
        Ok(Self { tokenizer, special_tokens })
    }

    /// Returns the tokenizer of this renderer.
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    /// Appends the block `header content_token payload <|end_message|>` to `rendering`.
    fn render_block(
        &self,
        rendering: &mut Rendering,
        header: &[TokenId],
        content_token: TokenId,
        payload: &str,
    ) -> Result<(), RendererError> {
        rendering.push_tokens(header);
        rendering.push_tokens(&[content_token]);
        rendering.push_text(&self.tokenizer, payload)?;
        rendering.push_tokens(&[self.special_tokens.end_message]);
        Ok(())
    }

    /// Appends the blocks of `message`, which belongs to `conversation`, to `rendering`.
    fn render_message(
        &self,
        rendering: &mut Rendering,
        conversation: &Conversation,
        message: &Message,
    ) -> Result<(), RendererError> {
        let special_tokens = &self.special_tokens;
        let role = message.role();
        if role != Role::Assistant {
            if message.reasoning().is_some_and(|reasoning| !reasoning.is_empty()) {
                return Err(RendererError::InvalidConversation {
                    message: format!("`{role}` messages cannot have reasoning"),
                });
            }
            if !message.tool_calls().is_empty() {
                return Err(RendererError::InvalidConversation {
                    message: format!("`{role}` messages cannot have tool calls"),
                });
            }
        }

        let (role_token, author) = match role {
            Role::System => (special_tokens.message_system, String::new()),
            Role::User => (special_tokens.message_user, String::new()),
            Role::Assistant => (special_tokens.message_model, message.name().unwrap_or_default().to_string()),
            Role::Tool => {
                let author = match (message.name(), message.tool_call_id()) {
                    (Some(name), _) if !name.is_empty() => name.to_string(),
                    (_, Some(tool_call_id)) if !tool_call_id.is_empty() => conversation
                        .messages()
                        .iter()
                        .filter(|message| message.role() == Role::Assistant)
                        .flat_map(Message::tool_calls)
                        .filter(|tool_call| tool_call.id() == Some(tool_call_id))
                        .map(ToolCall::name)
                        .collect(),
                    _ => String::new(),
                };
                (special_tokens.message_tool, author)
            }
        };
        let mut header = vec![role_token];
        header.extend(self.tokenizer.encode(&author, SpecialTokenHandling::Literal)?);

        rendering.set_supervised(role == Role::Assistant);
        let reasoning = message.reasoning().unwrap_or_default();
        if !reasoning.is_empty() {
            self.render_block(rendering, &header, special_tokens.content_thinking, reasoning)?;
        }
        let omit_content = matches!(message.content(), [ContentPart::Text(text)] if text.is_empty())
            && role == Role::Assistant
            && (!message.tool_calls().is_empty() || !reasoning.is_empty());
        if !omit_content {
            for part in message.content() {
                match part {
                    ContentPart::Text(text) => {
                        self.render_block(rendering, &header, special_tokens.content_text, text)?;
                    }
                    ContentPart::Image { token_count } => {
                        if !matches!(role, Role::User | Role::Tool) {
                            return Err(RendererError::InvalidConversation {
                                message: format!(
                                    "images are only supported in `user` and `tool` messages, not in `{role}` messages",
                                ),
                            });
                        }
                        rendering.push_tokens(&header);
                        rendering.push_tokens(&[special_tokens.content_image]);
                        rendering.push_image(special_tokens.image_placeholder, *token_count);
                        rendering.push_tokens(&[special_tokens.end_message]);
                    }
                }
            }
        }
        for tool_call in message.tool_calls() {
            if !author.is_empty() && author != tool_call.name() {
                return Err(RendererError::InvalidConversation {
                    message: format!(
                        "assistant name `{author}` does not match the function name `{}` of its tool call",
                        tool_call.name(),
                    ),
                });
            }
            // `ToolCall::new` validates that the arguments form an object.
            let JsonValue::Object(arguments) = tool_call.arguments() else { unreachable!() };
            let mut arguments = arguments.clone();
            arguments.sort_by(|(left, _), (right, _)| left.cmp(right));
            let payload = JsonValue::Object(vec![
                ("name".to_string(), JsonValue::String(tool_call.name().to_string())),
                ("args".to_string(), JsonValue::Object(arguments)),
            ]);
            let mut tool_call_header = vec![special_tokens.message_model];
            tool_call_header.extend(self.tokenizer.encode(tool_call.name(), SpecialTokenHandling::Literal)?);
            let payload = payload.to_json(false)?;
            self.render_block(rendering, &tool_call_header, special_tokens.content_invoke_tool_json, &payload)?;
        }
        if role == Role::Assistant {
            rendering.push_tokens(&[special_tokens.content_model_end_sampling]);
        }
        rendering.set_supervised(false);
        Ok(())
    }

    /// Returns a description of `token` for error messages.
    fn describe_token(&self, token: TokenId) -> String {
        match self.tokenizer.token(token) {
            Some(text) => format!("`{text}`"),
            None => format!("identifier {token}"),
        }
    }
}

impl<T: Tokenizer> Renderer for Inkling2Renderer<T> {
    fn render(&self, conversation: &Conversation) -> Result<RenderedConversation, RendererError> {
        let special_tokens = &self.special_tokens;
        let effort = thinking_effort(conversation.reasoning_effort())?;
        let mut rendering = Rendering::default();
        if !conversation.tools().is_empty() {
            let specifications = conversation
                .tools()
                .iter()
                .map(|tool| {
                    JsonValue::Object(vec![
                        ("description".to_string(), JsonValue::String(tool.description().to_string())),
                        ("name".to_string(), JsonValue::String(tool.name().to_string())),
                        ("parameters".to_string(), tool.parameters().clone()),
                        ("type".to_string(), JsonValue::String(tool.r#type().to_string())),
                    ])
                })
                .collect();
            let mut header = vec![special_tokens.message_system];
            header.extend(self.tokenizer.encode("tool_declare", SpecialTokenHandling::Literal)?);
            let payload = JsonValue::Array(specifications).to_json(true)?;
            self.render_block(&mut rendering, &header, special_tokens.content_xml, &payload)?;
        }
        let effort_header = [special_tokens.message_system];
        let mut effort_rendered = false;
        for message in conversation.messages() {
            if !effort_rendered && message.role() != Role::System {
                self.render_block(&mut rendering, &effort_header, special_tokens.content_text, &effort)?;
                effort_rendered = true;
            }
            self.render_message(&mut rendering, conversation, message)?;
        }
        if !effort_rendered {
            self.render_block(&mut rendering, &effort_header, special_tokens.content_text, &effort)?;
        }
        RenderedConversation::new(rendering.tokens, rendering.supervision, rendering.image_spans)
    }

    fn stop_tokens(&self) -> &[TokenId] {
        std::slice::from_ref(&self.special_tokens.content_model_end_sampling)
    }

    fn parse(&self, tokens: &[TokenId]) -> Result<Message, RendererError> {
        let special_tokens = &self.special_tokens;
        let mut message = Message::new(Role::Assistant);
        let mut reasoning = None::<String>;
        let mut position = 0;
        while position < tokens.len() {
            if tokens[position] == special_tokens.content_model_end_sampling {
                if let Some(token) = tokens.get(position + 1) {
                    return Err(RendererError::MalformedResponse {
                        message: format!(
                            "unexpected token {} at position {} after the stop token",
                            self.describe_token(*token),
                            position + 1,
                        ),
                    });
                }
                break;
            }
            let block_start = position;
            if tokens[position] == special_tokens.message_model {
                position += 1;
                while position < tokens.len() && !special_tokens.contains(tokens[position]) {
                    position += 1;
                }
            }
            let Some(content_token) = tokens.get(position).copied() else {
                return Err(RendererError::MalformedResponse {
                    message: format!("the block at position {block_start} ends before its content token"),
                });
            };
            if ![
                special_tokens.content_thinking,
                special_tokens.content_text,
                special_tokens.content_tool_error,
                special_tokens.content_invoke_tool_text,
                special_tokens.content_invoke_tool_json,
            ]
            .contains(&content_token)
            {
                return Err(RendererError::MalformedResponse {
                    message: format!(
                        "expected a content token at position {position} but found {}",
                        self.describe_token(content_token),
                    ),
                });
            }
            position += 1;
            let payload_start = position;
            while position < tokens.len() && !special_tokens.contains(tokens[position]) {
                position += 1;
            }
            match tokens.get(position) {
                Some(token) if *token == special_tokens.end_message => {}
                Some(token) => {
                    return Err(RendererError::MalformedResponse {
                        message: format!(
                            "expected `<|end_message|>` at position {position} but found {}",
                            self.describe_token(*token),
                        ),
                    });
                }
                None => {
                    return Err(RendererError::MalformedResponse {
                        message: format!("the block at position {block_start} ends without `<|end_message|>`"),
                    });
                }
            }
            let payload = self.tokenizer.decode(&tokens[payload_start..position], false)?;
            position += 1;
            if content_token == special_tokens.content_thinking {
                reasoning.get_or_insert_default().push_str(&payload);
            } else if content_token == special_tokens.content_invoke_tool_json {
                message = message.with_tool_call(parse_tool_call(&payload, block_start)?);
            } else {
                message = message.with_content_part(ContentPart::Text(payload));
            }
        }
        if let Some(reasoning) = reasoning {
            message = message.with_reasoning(reasoning);
        }
        Ok(message)
    }
}

/// Identifiers of the special tokens of the Inkling 2 format.
#[derive(Copy, Clone, Debug)]
struct Inkling2SpecialTokens {
    /// Role token of user messages, `<|message_user|>`.
    message_user: TokenId,

    /// Role token of assistant messages, `<|message_model|>`.
    message_model: TokenId,

    /// Role token of system messages, `<|message_system|>`.
    message_system: TokenId,

    /// Role token of tool messages, `<|message_tool|>`.
    message_tool: TokenId,

    /// Content token of text blocks, `<|content_text|>`.
    content_text: TokenId,

    /// Content token of image blocks, `<|content_image|>`.
    content_image: TokenId,

    /// Content token of reasoning blocks, `<|content_thinking|>`.
    content_thinking: TokenId,

    /// Content token of the tool declaration block, `<|content_xml|>`.
    content_xml: TokenId,

    /// Content token of JSON tool call blocks, `<|content_invoke_tool_json|>`.
    content_invoke_tool_json: TokenId,

    /// Content token of text tool call blocks, `<|content_invoke_tool_text|>`.
    content_invoke_tool_text: TokenId,

    /// Content token of tool error blocks, `<|content_tool_error|>`.
    content_tool_error: TokenId,

    /// Token that ends every block, `<|end_message|>`.
    end_message: TokenId,

    /// Token that ends assistant messages and stops sampling, `<|content_model_end_sampling|>`.
    content_model_end_sampling: TokenId,

    /// Image placeholder token, `<|unused_200054|>`.
    image_placeholder: TokenId,
}

impl Inkling2SpecialTokens {
    /// Returns whether `token` is one of these special tokens.
    fn contains(&self, token: TokenId) -> bool {
        [
            self.message_user,
            self.message_model,
            self.message_system,
            self.message_tool,
            self.content_text,
            self.content_image,
            self.content_thinking,
            self.content_xml,
            self.content_invoke_tool_json,
            self.content_invoke_tool_text,
            self.content_tool_error,
            self.end_message,
            self.content_model_end_sampling,
            self.image_placeholder,
        ]
        .contains(&token)
    }
}

/// Tokens of a [`RenderedConversation`] under construction.
#[derive(Clone, Debug, Default)]
struct Rendering {
    /// Tokens rendered so far.
    tokens: Vec<TokenId>,

    /// Supervision mask of [`Self::tokens`].
    supervision: Vec<bool>,

    /// Token ranges of the image placeholders rendered so far.
    image_spans: Vec<Range<usize>>,

    /// Whether the tokens that are appended next are supervised.
    supervised: bool,
}

impl Rendering {
    /// Sets whether the tokens that are appended next are supervised.
    fn set_supervised(&mut self, supervised: bool) {
        self.supervised = supervised;
    }

    /// Appends `tokens`.
    fn push_tokens(&mut self, tokens: &[TokenId]) {
        self.tokens.extend_from_slice(tokens);
        self.supervision.resize(self.tokens.len(), self.supervised);
    }

    /// Appends the tokens of `text`, encoded with [`SpecialTokenHandling::Literal`].
    fn push_text(&mut self, tokenizer: &impl Tokenizer, text: &str) -> Result<(), RendererError> {
        let tokens = tokenizer.encode(text, SpecialTokenHandling::Literal)?;
        self.push_tokens(&tokens);
        Ok(())
    }

    /// Appends `token_count` copies of `placeholder` and records their range as an image span.
    fn push_image(&mut self, placeholder: TokenId, token_count: usize) {
        let start = self.tokens.len();
        self.push_tokens(&vec![placeholder; token_count]);
        self.image_spans.push(start..self.tokens.len());
    }
}

/// Returns the text of the thinking effort block for `reasoning_effort`, where [`None`] selects the default effort.
fn thinking_effort(reasoning_effort: Option<&ReasoningEffort>) -> Result<String, RendererError> {
    let value = match reasoning_effort {
        None => 0.9,
        Some(ReasoningEffort::Named(name)) => match name.trim() {
            "none" => 0.0,
            "minimal" => 0.1,
            "low" => 0.2,
            "medium" => 0.7,
            "high" => 0.9,
            "xhigh" | "max" => 0.99,
            _ => {
                return Err(RendererError::InvalidConversation {
                    message: format!("unknown reasoning effort `{name}`"),
                });
            }
        },
        Some(ReasoningEffort::Value(value)) => *value,
    };
    if !(0.0..=0.99).contains(&value) {
        return Err(RendererError::InvalidConversation {
            message: format!("reasoning effort `{value}` is outside [0.0, 0.99]"),
        });
    }
    let value = if value == 0.0 { "0".to_string() } else { JsonValue::Float(value).to_json(false)? };
    Ok(format!("Thinking effort level: {value}"))
}

/// Parses the payload of a `<|content_invoke_tool_json|>` block that starts at token position `position`.
fn parse_tool_call(payload: &str, position: usize) -> Result<ToolCall, RendererError> {
    let value = JsonValue::parse(payload).map_err(|error| RendererError::MalformedResponse {
        message: format!("the payload of the tool call block at position {position} is not valid: {error}"),
    })?;
    match (&value, value.get("name"), value.get("args")) {
        (JsonValue::Object(entries), Some(JsonValue::String(name)), Some(arguments @ JsonValue::Object(_)))
            if entries.len() == 2 =>
        {
            ToolCall::new(name.clone(), arguments.clone())
        }
        _ => Err(RendererError::MalformedResponse {
            message: format!(
                "the payload of the tool call block at position {position} must be a JSON object with exactly a \
                string `name` and an object `args`",
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use pretty_assertions::assert_eq;

    use crate::json::JsonError;
    use crate::renderers::Tool;
    use crate::renderers::tests::TestTokenizer;
    use crate::tokenizers::{HuggingFaceTokenizer, TokenizerError};

    use super::*;

    /// Conversation with the tokens and the supervision mask that the chat template of the Inkling 2 Mini release
    /// produces for it with the reference tokenizer.
    struct ReleaseParityCase {
        /// Name of the case.
        name: &'static str,

        /// Conversation as JSON, in the form that the chat template takes.
        conversation: &'static str,

        /// Tokens of the rendered conversation.
        tokens: &'static [TokenId],

        /// Supervision of every token, as one `1` (supervised) or `0` character per token.
        supervision: &'static str,
    }

    /// Cases of the release parity test.
    const RELEASE_PARITY_CASES: &[ReleaseParityCase] = &[
        ReleaseParityCase {
            name: "single_turn",
            conversation: concat!(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"What is 2 + 2?\"},{\"role\":\"assistant\",\"content",
                "\":\"4\"}]}",
            ),
            tokens: &[
                200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010, 200000, 200004, 1775, 149884,
                220, 17, 196986, 220, 17, 30, 200010, 200001, 200004, 19, 200010, 200006,
            ],
            supervision: "0000000000000000000000011111",
        },
        ReleaseParityCase {
            name: "system_and_multi_turn",
            conversation: concat!(
                "{\"messages\":[{\"role\":\"system\",\"content\":\"You are terse.\"},{\"role\":\"user\",\"content\":",
                "\"Name a prime.\"},{\"role\":\"assistant\",\"content\":\"7\"},{\"role\":\"user\",\"content\":\"Anoth",
                "er.\"},{\"role\":\"assistant\",\"content\":\"11\"}]}",
            ),
            tokens: &[
                200002, 200004, 1735, 149914, 172044, 13, 200010, 200002, 200004, 34182, 279, 154554, 150628, 25, 220,
                15, 13, 24, 200010, 200000, 200004, 1353, 149878, 152200, 13, 200010, 200001, 200004, 22, 200010,
                200006, 200000, 200004, 83355, 13, 200010, 200001, 200004, 68052, 200010, 200006,
            ],
            supervision: "00000000000000000000000000111110000011111",
        },
        ReleaseParityCase {
            name: "generation_prompt",
            conversation: "{\"messages\":[{\"role\":\"user\",\"content\":\"Tell me a joke.\"}]}",
            tokens: &[
                200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010, 200000, 200004, 34751, 149991,
                149878, 193279, 13, 200010,
            ],
            supervision: "00000000000000000000",
        },
        ReleaseParityCase {
            name: "reasoning",
            conversation: concat!(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"Is 91 prime?\"},{\"role\":\"assistant\",\"reasoning_",
                "content\":\"91 = 7 \\u00d7 13.\",\"content\":\"No, 91 = 7 \\u00d7 13.\"}],\"reasoning_effort\":\"low",
                "\"}",
            ),
            tokens: &[
                200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 17, 200010, 200000, 200004, 1583, 220,
                70631, 152200, 30, 200010, 200001, 200008, 70631, 196977, 220, 22, 150766, 220, 68079, 13, 200010,
                200001, 200004, 982, 11, 220, 70631, 196977, 220, 22, 150766, 220, 68079, 13, 200010, 200006,
            ],
            supervision: "0000000000000000000011111111111111111111111111",
        },
        ReleaseParityCase {
            name: "effort_none",
            conversation: concat!(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"content\":\"Hello!\"",
                "}],\"reasoning_effort\":\"none\"}",
            ),
            tokens: &[
                200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 200010, 200000, 200004, 4896, 200010, 200001,
                200004, 32269, 0, 200010, 200006,
            ],
            supervision: "00000000000000111111",
        },
        ReleaseParityCase {
            name: "effort_number",
            conversation: concat!(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"content\":\"Hello!\"",
                "}],\"reasoning_effort\":0.35}",
            ),
            tokens: &[
                200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 68337, 200010, 200000, 200004, 4896,
                200010, 200001, 200004, 32269, 0, 200010, 200006,
            ],
            supervision: "0000000000000000111111",
        },
        ReleaseParityCase {
            name: "tool_call_round_trip",
            conversation: concat!(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"Weather in Paris for 3 days?\"},{\"role\":\"assistan",
                "t\",\"content\":\"\",\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\"",
                ":\"get_weather\",\"arguments\":{\"days\":3,\"city\":\"Paris\"}}}]},{\"role\":\"tool\",\"tool_call_id",
                "\":\"call_1\",\"content\":\"{\\\"forecast\\\":[\\\"sun\\\",\\\"rain\\\",\\\"sun\\\"]}\"},{\"role\":",
                "\"assistant\",\"content\":\"Sun, then rain, then sun.\"}],\"tools\":[{\"type\":\"function\",\"functi",
                "on\":{\"name\":\"get_weather\",\"description\":\"Returns the <b>forecast</b> for a city's weather & ",
                "more \\u2014 pr\\u00e9cis.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":",
                "\"string\"},\"days\":{\"type\":\"integer\"}},\"required\":[\"city\"]}}}]}",
            ),
            tokens: &[
                200002, 2251, 62, 76529, 200024, 58, 90, 1, 70469, 1, 25, 1, 80116, 149876, 196982, 65, 29, 92058, 27,
                14, 65, 29, 149887, 149878, 152652, 6, 82, 154046, 196992, 150033, 150220, 163113, 13, 1, 11, 1, 930,
                1, 25, 1, 832, 62, 30847, 1, 11, 1, 71968, 1, 25, 90, 1, 71961, 1, 25, 90, 1, 2958, 1, 25, 90, 1, 1008,
                1, 25, 1, 1485, 1, 92, 11, 1, 70095, 1, 25, 90, 1, 1008, 1, 25, 1, 4252, 1, 92, 92, 11, 1, 69812, 1,
                25, 58, 1, 2958, 1, 60, 11, 1, 1008, 1, 25, 1, 3156, 1, 92, 11, 1, 1008, 1, 25, 1, 36404, 1, 92, 60,
                200010, 200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010, 200000, 200004, 35120,
                149881, 152717, 149887, 220, 18, 151135, 30, 200010, 200001, 832, 62, 30847, 200049, 90, 1, 930, 1, 25,
                1, 832, 62, 30847, 1, 11, 1, 70747, 1, 25, 90, 1, 2958, 1, 25, 1, 3967, 1, 11, 1, 70095, 1, 25, 18, 92,
                92, 200010, 200006, 200003, 832, 62, 30847, 200004, 90, 1, 92058, 1, 25, 58, 1, 3705, 1, 11, 1, 1650,
                1, 11, 1, 3705, 1, 60, 92, 200010, 200001, 200004, 3744, 11, 150126, 154005, 11, 150126, 151483, 13,
                200010, 200006,
            ],
            supervision: concat!(
                "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
                "0000000000000000000000000000000000001111111111111111111111111111111111111100000000000000000000000001",
                "11111111111",
            ),
        },
        ReleaseParityCase {
            name: "tool_call_with_reasoning_and_text",
            conversation: concat!(
                "{\"messages\":[{\"role\":\"system\",\"content\":\"Use tools when needed.\"},{\"role\":\"user\",\"con",
                "tent\":\"Check Tokyo.\"},{\"role\":\"assistant\",\"reasoning_content\":\"I should call the tool.\",",
                "\"content\":\"Checking now.\",\"tool_calls\":[{\"id\":\"call_a\",\"type\":\"function\",\"function\":",
                "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Tokyo\",\"ratio\":0.5,\"nested\":{\"b\":[1,2],\"",
                "a\":null},\"flag\":true}}}]},{\"role\":\"tool\",\"name\":\"get_weather\",\"content\":\"cloudy\"},{\"",
                "role\":\"assistant\",\"content\":\"It is cloudy in Tokyo.\"}],\"tools\":[{\"type\":\"function\",\"fu",
                "nction\":{\"name\":\"get_weather\",\"description\":\"Returns the <b>forecast</b> for a city's weathe",
                "r & more \\u2014 pr\\u00e9cis.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"typ",
                "e\":\"string\"},\"days\":{\"type\":\"integer\"}},\"required\":[\"city\"]}}}]}",
            ),
            tokens: &[
                200002, 2251, 62, 76529, 200024, 58, 90, 1, 70469, 1, 25, 1, 80116, 149876, 196982, 65, 29, 92058, 27,
                14, 65, 29, 149887, 149878, 152652, 6, 82, 154046, 196992, 150033, 150220, 163113, 13, 1, 11, 1, 930,
                1, 25, 1, 832, 62, 30847, 1, 11, 1, 71968, 1, 25, 90, 1, 71961, 1, 25, 90, 1, 2958, 1, 25, 90, 1, 1008,
                1, 25, 1, 1485, 1, 92, 11, 1, 70095, 1, 25, 90, 1, 1008, 1, 25, 1, 4252, 1, 92, 92, 11, 1, 69812, 1,
                25, 58, 1, 2958, 1, 60, 11, 1, 1008, 1, 25, 1, 3156, 1, 92, 11, 1, 1008, 1, 25, 1, 36404, 1, 92, 60,
                200010, 200002, 200004, 1795, 152804, 150075, 151835, 13, 200010, 200002, 200004, 34182, 279, 154554,
                150628, 25, 220, 15, 13, 24, 200010, 200000, 200004, 3598, 182416, 13, 200010, 200001, 200008, 40,
                150201, 151272, 149876, 151955, 13, 200010, 200001, 200004, 3598, 279, 150386, 13, 200010, 200001, 832,
                62, 30847, 200049, 90, 1, 930, 1, 25, 1, 832, 62, 30847, 1, 11, 1, 70747, 1, 25, 90, 1, 2958, 1, 25, 1,
                34664, 1, 11, 1, 3385, 1, 25, 1750, 11, 1, 112324, 1, 25, 90, 1, 65, 1, 25, 58, 16, 11, 17, 60, 11, 1,
                64, 1, 25, 2451, 92, 11, 1, 34981, 1, 25, 15, 13, 20, 92, 92, 200010, 200006, 200003, 832, 62, 30847,
                200004, 365, 275, 1971, 200010, 200001, 200004, 1256, 149884, 151731, 275, 1971, 149881, 182416, 13,
                200010, 200006,
            ],
            supervision: concat!(
                "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
                "0000000000000000000000000000000000000001111111111111111111111111111111111111111111111111111111111111",
                "11111111111111111111111000000000111111111111",
            ),
        },
        ReleaseParityCase {
            name: "image_and_text_parts",
            conversation: concat!(
                "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this image.\"}",
                ",{\"type\":\"image\"}]},{\"role\":\"assistant\",\"content\":\"A cat on a mat.\"}]}",
            ),
            tokens: &[
                200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010, 200000, 200004, 30777, 39891,
                149928, 150641, 13, 200010, 200000, 200005, 200054, 200010, 200001, 200004, 32, 151683, 149894, 149878,
                151131, 13, 200010, 200006,
            ],
            supervision: "0000000000000000000000001111111111",
        },
        ReleaseParityCase {
            name: "empty_user_message",
            conversation: concat!(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"\"},{\"role\":\"assistant\",\"content\":\"You sent n",
                "othing.\"}]}",
            ),
            tokens: &[
                200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010, 200000, 200004, 200010,
                200001, 200004, 1735, 150678, 153575, 13, 200010, 200006,
            ],
            supervision: "00000000000000011111111",
        },
        ReleaseParityCase {
            name: "assistant_name_and_float_arguments",
            conversation: concat!(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"Scale it.\"},{\"role\":\"assistant\",\"name\":\"scal",
                "e\",\"content\":\"\",\"tool_calls\":[{\"type\":\"function\",\"function\":{\"name\":\"scale\",\"argum",
                "ents\":{\"z\":1.0,\"tiny\":1e-07,\"big\":1e+16,\"text\":\"a\\\"b\\\\c\\n\\u00e9\\u0001\",\"list\":[1",
                ".5,-2,false,null]}}}]},{\"role\":\"tool\",\"name\":\"scale\",\"content\":[{\"type\":\"text\",\"text",
                "\":\"done\"},{\"type\":\"text\",\"text\":\"really\"}]}],\"tools\":[{\"name\":\"scale\",\"parameters",
                "\":{\"type\":\"object\"}}],\"reasoning_effort\":\"xhigh\"}",
            ),
            tokens: &[
                200002, 2251, 62, 76529, 200024, 58, 90, 1, 70469, 1, 25, 68723, 11, 1, 930, 1, 25, 1, 2275, 1, 11, 1,
                71968, 1, 25, 90, 1, 1008, 1, 25, 1, 3156, 1, 92, 11, 1, 1008, 1, 25, 1, 36404, 1, 92, 60, 200010,
                200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 68436, 200010, 200000, 200004, 31448,
                149921, 13, 200010, 200001, 2275, 200049, 90, 1, 930, 1, 25, 1, 2275, 1, 11, 1, 70747, 1, 25, 90, 1,
                3197, 1, 25, 16, 68, 10, 68080, 11, 1, 1229, 1, 25, 58, 16, 13, 20, 11, 12, 17, 11, 2121, 11, 2451, 60,
                11, 1, 911, 1, 25, 1, 64, 59, 1, 65, 68566, 66, 59, 68229, 59, 84, 68128, 16, 1, 11, 1, 31457, 1, 25,
                16, 68, 12, 68620, 11, 1, 89, 1, 25, 16, 13, 15, 92, 92, 200010, 200006, 200003, 2275, 200004, 2503,
                200010, 200003, 2275, 200004, 70866, 200010,
            ],
            supervision: concat!(
                "0000000000000000000000000000000000000000000000000000000000000001111111111111111111111111111111111111",
                "1111111111111111111111111111111111111111111110000000000",
            ),
        },
        ReleaseParityCase {
            name: "only_system",
            conversation: "{\"messages\":[{\"role\":\"system\",\"content\":\"Be helpful.\"}]}",
            tokens: &[
                200002, 200004, 1618, 160385, 13, 200010, 200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13,
                24, 200010,
            ],
            supervision: "000000000000000000",
        },
    ];

    /// Special tokens of the [`TestTokenizer`] of these tests, whose identifiers are 256 onward in this order.
    const SPECIAL_TOKENS: [&str; 14] = [
        "<|message_user|>",
        "<|message_model|>",
        "<|message_system|>",
        "<|message_tool|>",
        "<|content_text|>",
        "<|content_image|>",
        "<|content_thinking|>",
        "<|content_xml|>",
        "<|content_invoke_tool_json|>",
        "<|content_invoke_tool_text|>",
        "<|content_tool_error|>",
        "<|end_message|>",
        "<|content_model_end_sampling|>",
        "<|unused_200054|>",
    ];

    /// Thinking effort block for the default effort.
    const DEFAULT_EFFORT: &str = "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>";

    /// Returns an [`Inkling2Renderer`] that uses a [`TestTokenizer`] with [`SPECIAL_TOKENS`].
    fn renderer() -> Inkling2Renderer<TestTokenizer> {
        Inkling2Renderer::new(TestTokenizer::new(&SPECIAL_TOKENS)).unwrap()
    }

    /// Returns the tokens of `text` with special token strings parsed, using the [`TestTokenizer`] of [`renderer`].
    fn tokens(text: &str) -> Vec<TokenId> {
        renderer().tokenizer().encode(text, SpecialTokenHandling::Parse).unwrap()
    }

    /// Returns the expected rendering of `segments`, which are texts in which special token strings are parsed, each
    /// paired with whether its tokens are supervised, with the image spans `image_spans`.
    fn rendered(
        segments: &[(&str, bool)],
        image_spans: Vec<Range<usize>>,
    ) -> Result<RenderedConversation, RendererError> {
        let mut all_tokens = Vec::new();
        let mut supervision = Vec::new();
        for (text, supervised) in segments {
            let segment_tokens = tokens(text);
            supervision.extend(std::iter::repeat_n(*supervised, segment_tokens.len()));
            all_tokens.extend(segment_tokens);
        }
        RenderedConversation::new(all_tokens, supervision, image_spans)
    }

    /// Returns an object [`JsonValue`] with `entries`.
    fn object(entries: Vec<(&str, JsonValue)>) -> JsonValue {
        JsonValue::Object(entries.into_iter().map(|(key, value)| (key.to_string(), value)).collect())
    }

    /// Returns the string of `value`, which must be a [`JsonValue::String`], or [`None`] if it is missing or empty.
    fn fixture_string(value: Option<&JsonValue>) -> Option<String> {
        match value {
            Some(JsonValue::String(text)) if !text.is_empty() => Some(text.clone()),
            None | Some(JsonValue::Null | JsonValue::String(_)) => None,
            Some(value) => panic!("expected a string but found {value:?}"),
        }
    }

    /// Converts an OpenAI style conversation of the release fixtures into a [`Conversation`], treating empty values
    /// like the chat template does.
    fn fixture_conversation(fixture: &JsonValue) -> Conversation {
        let Some(JsonValue::Array(fixture_messages)) = fixture.get("messages") else {
            panic!("expected a `messages` array");
        };
        let mut messages = Vec::new();
        for fixture_message in fixture_messages {
            let role = match fixture_string(fixture_message.get("role")).as_deref() {
                Some("system") => Role::System,
                Some("user") => Role::User,
                Some("assistant") => Role::Assistant,
                Some("tool") => Role::Tool,
                role => panic!("unexpected role {role:?}"),
            };
            let mut message = Message::new(role);
            match fixture_message.get("content") {
                None | Some(JsonValue::Null) => {}
                Some(JsonValue::String(text)) => message = message.with_content_part(ContentPart::Text(text.clone())),
                Some(JsonValue::Array(parts)) => {
                    for part in parts {
                        let part = match fixture_string(part.get("type")).as_deref() {
                            Some("text") => ContentPart::Text(fixture_string(part.get("text")).unwrap_or_default()),
                            Some("image") => ContentPart::Image { token_count: 1 },
                            part_type => panic!("unexpected content part type {part_type:?}"),
                        };
                        message = message.with_content_part(part);
                    }
                }
                Some(content) => panic!("unexpected content {content:?}"),
            }
            if let Some(reasoning) = fixture_string(fixture_message.get("reasoning_content")) {
                message = message.with_reasoning(reasoning);
            }
            if let Some(name) = fixture_string(fixture_message.get("name")) {
                message = message.with_name(name);
            }
            if let Some(tool_call_id) = fixture_string(fixture_message.get("tool_call_id")) {
                message = message.with_tool_call_id(tool_call_id);
            }
            if let Some(JsonValue::Array(tool_calls)) = fixture_message.get("tool_calls") {
                for fixture_tool_call in tool_calls {
                    let function = fixture_tool_call.get("function").unwrap();
                    let arguments = match function.get("arguments") {
                        Some(arguments @ JsonValue::Object(_)) => arguments.clone(),
                        _ => JsonValue::Object(Vec::new()),
                    };
                    let mut tool_call =
                        ToolCall::new(fixture_string(function.get("name")).unwrap(), arguments).unwrap();
                    if let Some(id) = fixture_string(fixture_tool_call.get("id")) {
                        tool_call = tool_call.with_id(id);
                    }
                    message = message.with_tool_call(tool_call);
                }
            }
            messages.push(message);
        }
        let mut conversation = Conversation::new(messages);
        if let Some(JsonValue::Array(tools)) = fixture.get("tools") {
            for fixture_tool in tools {
                let function = fixture_tool.get("function").unwrap_or(fixture_tool);
                let mut tool = Tool::new(fixture_string(function.get("name")).unwrap());
                if let Some(description) = fixture_string(function.get("description")) {
                    tool = tool.with_description(description);
                }
                match function.get("parameters") {
                    Some(JsonValue::Object(entries)) if !entries.is_empty() => {
                        tool = tool.with_parameters(JsonValue::Object(entries.clone()));
                    }
                    None | Some(JsonValue::Null | JsonValue::Object(_)) => {}
                    Some(parameters) => panic!("unexpected parameters {parameters:?}"),
                }
                if let Some(r#type) = fixture_string(fixture_tool.get("type")) {
                    tool = tool.with_type(r#type);
                }
                conversation = conversation.with_tool(tool);
            }
        }
        match fixture.get("reasoning_effort") {
            None | Some(JsonValue::Null) => {}
            Some(JsonValue::String(name)) => {
                conversation = conversation.with_reasoning_effort(ReasoningEffort::Named(name.clone()));
            }
            Some(JsonValue::Float(value)) => {
                conversation = conversation.with_reasoning_effort(ReasoningEffort::Value(*value));
            }
            Some(JsonValue::Integer(value)) => {
                conversation = conversation.with_reasoning_effort(ReasoningEffort::Value(*value as f64));
            }
            Some(effort) => panic!("unexpected reasoning effort {effort:?}"),
        }
        conversation
    }

    #[test]
    fn test_inkling2_renderer_new() {
        let renderer = Inkling2Renderer::new(TestTokenizer::new(&SPECIAL_TOKENS)).unwrap();
        assert_eq!(renderer.stop_tokens(), &[268]);

        let special_tokens =
            SPECIAL_TOKENS.iter().copied().filter(|token| *token != "<|content_xml|>").collect::<Vec<_>>();
        assert!(matches!(
            Inkling2Renderer::new(TestTokenizer::new(&special_tokens)),
            Err(RendererError::MissingSpecialToken { token }) if token == "<|content_xml|>",
        ));
    }

    #[test]
    fn test_inkling2_renderer_tokenizer() {
        assert_eq!(renderer().tokenizer().vocabulary_size(), 270);
    }

    #[test]
    fn test_inkling2_renderer_render() {
        let conversation = Conversation::new(vec![
            Message::text(Role::System, "Be brief."),
            Message::text(Role::User, "Hi"),
            Message::text(Role::Assistant, "Hello!"),
            Message::text(Role::User, "Bye"),
        ]);
        assert_eq!(
            renderer().render(&conversation),
            rendered(
                &[
                    ("<|message_system|><|content_text|>Be brief.<|end_message|>", false),
                    (DEFAULT_EFFORT, false),
                    ("<|message_user|><|content_text|>Hi<|end_message|>", false),
                    ("<|message_model|><|content_text|>Hello!<|end_message|><|content_model_end_sampling|>", true),
                    ("<|message_user|><|content_text|>Bye<|end_message|>", false),
                ],
                Vec::new(),
            ),
        );
    }

    #[test]
    fn test_inkling2_renderer_render_thinking_effort() {
        let effort_block =
            |effort: &str| format!("<|message_system|><|content_text|>Thinking effort level: {effort}<|end_message|>");
        let render =
            |effort: ReasoningEffort| renderer().render(&Conversation::new(Vec::new()).with_reasoning_effort(effort));

        // Without messages, or with system messages only, the effort block ends the conversation.
        assert_eq!(renderer().render(&Conversation::new(Vec::new())), rendered(&[(DEFAULT_EFFORT, false)], Vec::new()));
        assert_eq!(
            renderer().render(&Conversation::new(vec![Message::text(Role::System, "Be helpful.")])),
            rendered(
                &[("<|message_system|><|content_text|>Be helpful.<|end_message|>", false), (DEFAULT_EFFORT, false)],
                Vec::new(),
            ),
        );

        // Named efforts map to fixed values, and numeric efforts are spelled like Python floats except for zero.
        let cases = [
            (ReasoningEffort::Named("none".to_string()), "0"),
            (ReasoningEffort::Named("minimal".to_string()), "0.1"),
            (ReasoningEffort::Named("low".to_string()), "0.2"),
            (ReasoningEffort::Named("medium".to_string()), "0.7"),
            (ReasoningEffort::Named(" high ".to_string()), "0.9"),
            (ReasoningEffort::Named("xhigh".to_string()), "0.99"),
            (ReasoningEffort::Named("max".to_string()), "0.99"),
            (ReasoningEffort::Value(0.0), "0"),
            (ReasoningEffort::Value(-0.0), "0"),
            (ReasoningEffort::Value(0.35), "0.35"),
            (ReasoningEffort::Value(0.99), "0.99"),
            (ReasoningEffort::Value(1e-5), "1e-05"),
        ];
        for (effort, expected) in cases {
            assert_eq!(render(effort.clone()), rendered(&[(&effort_block(expected), false)], Vec::new()), "{effort:?}");
        }

        assert!(matches!(
            render(ReasoningEffort::Named("extreme".to_string())),
            Err(RendererError::InvalidConversation { message }) if message == "unknown reasoning effort `extreme`",
        ));
        assert!(matches!(
            render(ReasoningEffort::Value(1.0)),
            Err(RendererError::InvalidConversation { message })
                if message == "reasoning effort `1` is outside [0.0, 0.99]",
        ));
        assert!(matches!(
            render(ReasoningEffort::Value(f64::NAN)),
            Err(RendererError::InvalidConversation { message })
                if message == "reasoning effort `NaN` is outside [0.0, 0.99]",
        ));
    }

    #[test]
    fn test_inkling2_renderer_render_tools() {
        let parameters = object(vec![
            ("type", JsonValue::String("object".to_string())),
            ("properties", object(vec![("city", object(vec![("type", JsonValue::String("string".to_string()))]))])),
        ]);
        let arguments = object(vec![
            ("days", JsonValue::Integer(3)),
            ("city", JsonValue::String("Paris".to_string())),
            (
                "options",
                object(vec![
                    ("units", JsonValue::String("metric".to_string())),
                    ("detail", JsonValue::Array(vec![JsonValue::Float(1.0), JsonValue::Null])),
                ]),
            ),
        ]);
        let conversation = Conversation::new(vec![
            Message::text(Role::User, "Paris?"),
            Message::text(Role::Assistant, "")
                .with_tool_call(ToolCall::new("get_weather", arguments).unwrap().with_id("call_1")),
            Message::text(Role::Tool, "sunny").with_tool_call_id("call_1"),
            Message::text(Role::Assistant, "Sunny."),
        ])
        .with_tool(Tool::new("get_weather").with_description("Weather — précis.").with_parameters(parameters));

        // Tool specifications are sorted recursively, while only the top level of tool arguments is sorted.
        assert_eq!(
            renderer().render(&conversation),
            rendered(
                &[
                    (
                        "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Weather — précis.\",\
                        \"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\
                        \"type\":\"object\"},\"type\":\"function\"}]<|end_message|>",
                        false,
                    ),
                    (DEFAULT_EFFORT, false),
                    ("<|message_user|><|content_text|>Paris?<|end_message|>", false),
                    (
                        "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":\
                        {\"city\":\"Paris\",\"days\":3,\"options\":{\"units\":\"metric\",\"detail\":[1.0,null]}}}\
                        <|end_message|><|content_model_end_sampling|>",
                        true,
                    ),
                    ("<|message_tool|>get_weather<|content_text|>sunny<|end_message|>", false),
                    ("<|message_model|><|content_text|>Sunny.<|end_message|><|content_model_end_sampling|>", true),
                ],
                Vec::new(),
            ),
        );
    }

    #[test]
    fn test_inkling2_renderer_render_tool_authors() {
        let tool_call = |name: &str| ToolCall::new(name, JsonValue::Object(Vec::new())).unwrap().with_id("call");
        let conversation = Conversation::new(vec![
            Message::new(Role::Assistant).with_tool_call(tool_call("f")).with_tool_call(tool_call("g")),
            Message::text(Role::Tool, "a").with_name("search").with_tool_call_id("call"),
            Message::text(Role::Tool, "b").with_tool_call_id("call"),
            Message::text(Role::Tool, "c").with_tool_call_id("other"),
            Message::text(Role::Tool, "d"),
        ]);

        // A name takes precedence over the tool call identifier, and the function names of all the tool calls with
        // the identifier are concatenated like the chat template does.
        assert_eq!(
            renderer().render(&conversation),
            rendered(
                &[
                    (DEFAULT_EFFORT, false),
                    (
                        "<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}<|end_message|>\
                        <|message_model|>g<|content_invoke_tool_json|>{\"name\":\"g\",\"args\":{}}<|end_message|>\
                        <|content_model_end_sampling|>",
                        true,
                    ),
                    ("<|message_tool|>search<|content_text|>a<|end_message|>", false),
                    ("<|message_tool|>fg<|content_text|>b<|end_message|>", false),
                    ("<|message_tool|><|content_text|>c<|end_message|>", false),
                    ("<|message_tool|><|content_text|>d<|end_message|>", false),
                ],
                Vec::new(),
            ),
        );
    }

    #[test]
    fn test_inkling2_renderer_render_assistant_messages() {
        let conversation = Conversation::new(vec![
            Message::text(Role::User, ""),
            Message::text(Role::Assistant, "").with_reasoning("Think."),
            Message::new(Role::Assistant),
            Message::text(Role::Assistant, "Hi").with_name("helper").with_reasoning(""),
            Message::text(Role::Assistant, ""),
        ]);

        // An empty text part is an empty block, except as the only content of an assistant message that has
        // reasoning or tool calls, and every assistant message ends with the stop token.
        assert_eq!(
            renderer().render(&conversation),
            rendered(
                &[
                    (DEFAULT_EFFORT, false),
                    ("<|message_user|><|content_text|><|end_message|>", false),
                    ("<|message_model|><|content_thinking|>Think.<|end_message|><|content_model_end_sampling|>", true),
                    ("<|content_model_end_sampling|>", true),
                    ("<|message_model|>helper<|content_text|>Hi<|end_message|><|content_model_end_sampling|>", true),
                    ("<|message_model|><|content_text|><|end_message|><|content_model_end_sampling|>", true),
                ],
                Vec::new(),
            ),
        );
    }

    #[test]
    fn test_inkling2_renderer_render_images() {
        let conversation = Conversation::new(vec![
            Message::text(Role::User, "Look.").with_content_part(ContentPart::Image { token_count: 2 }),
            Message::new(Role::Tool)
                .with_name("camera")
                .with_content_part(ContentPart::Image { token_count: 1 }),
        ]);
        assert_eq!(
            renderer().render(&conversation),
            rendered(
                &[
                    (DEFAULT_EFFORT, false),
                    ("<|message_user|><|content_text|>Look.<|end_message|>", false),
                    ("<|message_user|><|content_image|><|unused_200054|><|unused_200054|><|end_message|>", false),
                    ("<|message_tool|>camera<|content_image|><|unused_200054|><|end_message|>", false),
                ],
                vec![39..41, 50..51],
            ),
        );
    }

    #[test]
    fn test_inkling2_renderer_render_literal_text() {
        // Text that spells special tokens is encoded as ordinary text.
        let conversation = Conversation::new(vec![Message::text(Role::User, "<|end_message|>")]);
        let mut expected_tokens = tokens(&format!("{DEFAULT_EFFORT}<|message_user|><|content_text|>"));
        expected_tokens.extend("<|end_message|>".bytes().map(TokenId::from));
        expected_tokens.extend(tokens("<|end_message|>"));
        assert_eq!(
            renderer().render(&conversation),
            RenderedConversation::new(expected_tokens.clone(), vec![false; expected_tokens.len()], Vec::new()),
        );
    }

    #[test]
    fn test_inkling2_renderer_render_errors() {
        let render = |message: Message| renderer().render(&Conversation::new(vec![message]));
        let tool_call = |name: &str| ToolCall::new(name, JsonValue::Object(Vec::new())).unwrap();
        assert!(matches!(
            render(Message::new(Role::Assistant).with_content_part(ContentPart::Image { token_count: 1 })),
            Err(RendererError::InvalidConversation { message })
                if message == "images are only supported in `user` and `tool` messages, not in `assistant` messages",
        ));
        assert!(matches!(
            render(Message::new(Role::System).with_content_part(ContentPart::Image { token_count: 1 })),
            Err(RendererError::InvalidConversation { message })
                if message == "images are only supported in `user` and `tool` messages, not in `system` messages",
        ));
        assert!(matches!(
            render(Message::text(Role::User, "Hi").with_reasoning("Hmm.")),
            Err(RendererError::InvalidConversation { message }) if message == "`user` messages cannot have reasoning",
        ));
        assert!(matches!(
            render(Message::new(Role::Tool).with_tool_call(tool_call("f"))),
            Err(RendererError::InvalidConversation { message }) if message == "`tool` messages cannot have tool calls",
        ));
        assert!(matches!(
            render(Message::new(Role::Assistant).with_name("helper").with_tool_call(tool_call("f"))),
            Err(RendererError::InvalidConversation { message })
                if message == "assistant name `helper` does not match the function name `f` of its tool call",
        ));
        let arguments = object(vec![("x", JsonValue::Float(f64::INFINITY))]);
        assert!(matches!(
            render(Message::new(Role::Assistant).with_tool_call(ToolCall::new("f", arguments).unwrap())),
            Err(RendererError::Json(JsonError::Invalid { message }))
                if message == "cannot serialize the non-finite float `inf`",
        ));
    }

    #[test]
    #[ignore = "requires the Inkling 2 release directory in `RYFT_MODELS_INKLING2_RELEASE_DIRECTORY`"]
    fn test_inkling2_renderer_render_release_parity() {
        let directory = std::env::var("RYFT_MODELS_INKLING2_RELEASE_DIRECTORY").unwrap();
        let tokenizer = HuggingFaceTokenizer::from_file(Path::new(&directory).join("tokenizer.json")).unwrap();
        let renderer = Inkling2Renderer::new(tokenizer).unwrap();
        assert_eq!(RELEASE_PARITY_CASES.len(), 12);
        for case in RELEASE_PARITY_CASES {
            let name = case.name;
            let expected_supervision = case.supervision.chars().map(|character| character == '1').collect::<Vec<_>>();
            let conversation = fixture_conversation(&JsonValue::parse(case.conversation).unwrap());
            let rendered = renderer.render(&conversation).unwrap();
            assert_eq!(rendered.tokens(), case.tokens, "{name}");
            assert_eq!(rendered.supervision(), expected_supervision.as_slice(), "{name}");
        }
    }

    #[test]
    fn test_inkling2_renderer_stop_tokens() {
        let renderer = renderer();
        assert_eq!(renderer.stop_tokens(), &[renderer.tokenizer().token_id("<|content_model_end_sampling|>").unwrap()]);
    }

    #[test]
    fn test_inkling2_renderer_parse() {
        let renderer = renderer();

        // Rendering and then parsing an assistant message round trips when its top level argument keys are sorted.
        let arguments = object(vec![
            ("city", JsonValue::String("Tokyo".to_string())),
            ("nested", object(vec![("b", JsonValue::Array(vec![JsonValue::Integer(1)])), ("a", JsonValue::Null)])),
        ]);
        let message = Message::text(Role::Assistant, "Checking.")
            .with_reasoning("Let me check.")
            .with_tool_call(ToolCall::new("get_weather", arguments).unwrap());
        let rendered = renderer.render(&Conversation::new(vec![message.clone()])).unwrap();
        let sampled = rendered
            .tokens()
            .iter()
            .zip(rendered.supervision())
            .filter(|(_, supervised)| **supervised)
            .map(|(token, _)| *token)
            .collect::<Vec<_>>();
        assert_eq!(renderer.parse(&sampled), Ok(message.clone()));
        assert_eq!(renderer.parse(&sampled[..sampled.len() - 1]), Ok(message));

        assert_eq!(renderer.parse(&[]), Ok(Message::new(Role::Assistant)));
        assert_eq!(renderer.parse(&tokens("<|content_model_end_sampling|>")), Ok(Message::new(Role::Assistant)));
    }

    #[test]
    fn test_inkling2_renderer_parse_lenient_blocks() {
        // Blocks may omit `<|message_model|>`, author names are ignored, and tool errors and text tool calls are text.
        assert_eq!(
            renderer().parse(&tokens(
                "<|content_text|>Hi<|end_message|><|message_model|>helper<|content_tool_error|>oops<|end_message|>\
                <|content_invoke_tool_text|>f()<|end_message|><|content_thinking|>a<|end_message|>\
                <|content_thinking|>b<|end_message|>",
            )),
            Ok(Message::text(Role::Assistant, "Hi")
                .with_content_part(ContentPart::Text("oops".to_string()))
                .with_content_part(ContentPart::Text("f()".to_string()))
                .with_reasoning("ab")),
        );
    }

    #[test]
    fn test_inkling2_renderer_parse_errors() {
        let parse = |text: &str| renderer().parse(&tokens(text));
        let cases = [
            ("<|content_model_end_sampling|>x", "unexpected token `x` at position 1 after the stop token"),
            ("<|message_model|>name", "the block at position 0 ends before its content token"),
            ("<|message_model|><|end_message|>", "expected a content token at position 1 but found `<|end_message|>`"),
            ("<|content_xml|>x<|end_message|>", "expected a content token at position 0 but found `<|content_xml|>`"),
            ("<|content_text|>Hi", "the block at position 0 ends without `<|end_message|>`"),
            (
                "<|content_text|>Hi<|content_text|>",
                "expected `<|end_message|>` at position 3 but found `<|content_text|>`",
            ),
            (
                "<|content_invoke_tool_json|>{<|end_message|>",
                "the payload of the tool call block at position 0 is not valid: invalid JSON: unexpected end of input \
                at byte 1",
            ),
            (
                "<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":[]}<|end_message|>",
                "the payload of the tool call block at position 0 must be a JSON object with exactly a string `name` \
                and an object `args`",
            ),
            (
                "<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{},\"id\":\"x\"}<|end_message|>",
                "the payload of the tool call block at position 0 must be a JSON object with exactly a string `name` \
                and an object `args`",
            ),
        ];
        for (text, expected_message) in cases {
            assert!(
                matches!(
                    parse(text),
                    Err(RendererError::MalformedResponse { message }) if message == expected_message,
                ),
                "{text:?}",
            );
        }
        assert!(matches!(
            renderer().parse(&[260, 999, 267]),
            Err(RendererError::Tokenizer(TokenizerError::Decoding { message }))
                if message == "unknown token identifier 999",
        ));
    }
}
