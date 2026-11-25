//! [`JsonValue`], a JSON value whose objects preserve the order of their keys, together with a strict parser and a
//! compact serialization that spells JSON exactly like Python's `json` module. Renderers need that spelling, because
//! chat templates embed JSON in conversations, and configuration readers use the parser because it converts numbers
//! to the nearest `f64`.

use thiserror::Error;

// TODO(eaplatanios): Review this module.

/// Error returned when JSON text cannot be parsed or when a [`JsonValue`] cannot be serialized.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum JsonError {
    #[error("invalid JSON: {message}")]
    Invalid { message: String },
}

/// JSON value whose objects preserve the order of their keys.
///
/// Chat templates spell JSON with Python's `json.dumps`, whose output depends on the insertion order of object keys
/// and on whether a number was parsed as an integer or as a float, so this type retains both. [`JsonValue::parse`]
/// follows Python's `json.loads` in these respects, and [`JsonValue::to_json`] spells values like
/// `json.dumps(value, ensure_ascii=False, separators=(",", ":"))`.
#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    /// The `null` literal.
    Null,

    /// The `true` or `false` literal.
    Boolean(bool),

    /// Number without a fraction or an exponent (e.g., `-2`).
    Integer(i64),

    /// Number with a fraction or an exponent (e.g., `1.0` or `1e-07`).
    Float(f64),

    /// String.
    String(String),

    /// Array of values.
    Array(Vec<JsonValue>),

    /// Object, as its key and value pairs in insertion order.
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    /// Parses `json`, which must contain exactly one JSON value surrounded by optional whitespace.
    ///
    /// The parser is strict. It rejects the `NaN` and `Infinity` extensions, numbers whose magnitude overflows to an
    /// infinite float, integers that do not fit in an [`i64`], unpaired surrogate escapes, unescaped control characters
    /// in strings, and nesting deeper than 128 levels. Like Python's `json.loads`, a repeated object key keeps the
    /// position of its first occurrence and the value of its last occurrence.
    pub fn parse(json: &str) -> Result<Self, JsonError> {
        let mut parser = JsonParser { json, position: 0 };
        parser.skip_whitespace();
        let value = parser.parse_value(0)?;
        parser.skip_whitespace();
        if parser.position < json.len() {
            return Err(parser.unexpected_character());
        }
        Ok(value)
    }

    /// Returns the value of the first entry with `key` if this is an object, and [`None`] otherwise.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            Self::Object(entries) => entries.iter().find(|(entry_key, _)| entry_key == key).map(|(_, value)| value),
            _ => None,
        }
    }

    /// Serializes this value compactly (i.e., without whitespace), spelled like Python's `json.dumps` with
    /// `ensure_ascii=False` and `separators=(",", ":")`. Strings escape `"`, `\`, and the control characters below
    /// `U+0020`, and leave all other characters unescaped. Floats use the shortest digits that round trip, with a
    /// trailing `.0` for integral values and scientific notation (e.g., `1e-07` and `1e+16`) when their decimal
    /// exponent is below -4 or at least 16. Serialization fails for non-finite floats.
    ///
    /// # Parameters
    ///
    ///   - `sort_keys`: Whether the keys of all objects, including nested ones, are sorted by code point instead of
    ///     kept in insertion order.
    pub fn to_json(&self, sort_keys: bool) -> Result<String, JsonError> {
        let mut json = String::new();
        self.write_json(sort_keys, &mut json)?;
        Ok(json)
    }

    /// Appends the serialization of [`JsonValue::to_json`] to `json`.
    fn write_json(&self, sort_keys: bool, json: &mut String) -> Result<(), JsonError> {
        match self {
            Self::Null => json.push_str("null"),
            Self::Boolean(value) => json.push_str(if *value { "true" } else { "false" }),
            Self::Integer(value) => json.push_str(&value.to_string()),
            Self::Float(value) => json.push_str(&format_float(*value)?),
            Self::String(value) => write_string(value, json),
            Self::Array(values) => {
                json.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        json.push(',');
                    }
                    value.write_json(sort_keys, json)?;
                }
                json.push(']');
            }
            Self::Object(entries) => {
                let mut entries = entries.iter().collect::<Vec<_>>();
                if sort_keys {
                    entries.sort_by(|(left, _), (right, _)| left.cmp(right));
                }
                json.push('{');
                for (index, (key, value)) in entries.into_iter().enumerate() {
                    if index > 0 {
                        json.push(',');
                    }
                    write_string(key, json);
                    json.push(':');
                    value.write_json(sort_keys, json)?;
                }
                json.push('}');
            }
        }
        Ok(())
    }
}

impl From<serde_json::Value> for JsonValue {
    fn from(value: serde_json::Value) -> Self {
        // Objects keep the iteration order of `serde_json::Map`, which is sorted unless the `preserve_order` feature
        // of `serde_json` is enabled. Unsigned integers above `i64::MAX` become the nearest float.
        match value {
            serde_json::Value::Null => Self::Null,
            serde_json::Value::Bool(value) => Self::Boolean(value),
            serde_json::Value::Number(number) => match number.as_i64() {
                Some(value) => Self::Integer(value),
                None => Self::Float(number.as_f64().unwrap()),
            },
            serde_json::Value::String(value) => Self::String(value),
            serde_json::Value::Array(values) => Self::Array(values.into_iter().map(Self::from).collect()),
            serde_json::Value::Object(entries) => {
                Self::Object(entries.into_iter().map(|(key, value)| (key, Self::from(value))).collect())
            }
        }
    }
}

/// Maximum nesting depth of arrays and objects that [`JsonValue::parse`] accepts.
const MAXIMUM_DEPTH: usize = 128;

/// Recursive descent parser state of [`JsonValue::parse`].
struct JsonParser<'o> {
    /// Text being parsed.
    json: &'o str,

    /// Byte offset of the next unparsed character of [`Self::json`].
    position: usize,
}

impl JsonParser<'_> {
    /// Parses the value at the current position, which is nested in `depth` arrays and objects.
    fn parse_value(&mut self, depth: usize) -> Result<JsonValue, JsonError> {
        match self.peek() {
            Some(b'n') => self.parse_literal("null", JsonValue::Null),
            Some(b't') => self.parse_literal("true", JsonValue::Boolean(true)),
            Some(b'f') => self.parse_literal("false", JsonValue::Boolean(false)),
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(b'[' | b'{') if depth == MAXIMUM_DEPTH => {
                Err(self.error(format!("nesting exceeds {MAXIMUM_DEPTH} levels")))
            }
            Some(b'[') => {
                self.position += 1;
                let mut values = Vec::new();
                self.skip_whitespace();
                if self.peek() == Some(b']') {
                    self.position += 1;
                    return Ok(JsonValue::Array(values));
                }
                loop {
                    self.skip_whitespace();
                    values.push(self.parse_value(depth + 1)?);
                    self.skip_whitespace();
                    match self.peek() {
                        Some(b',') => self.position += 1,
                        Some(b']') => {
                            self.position += 1;
                            return Ok(JsonValue::Array(values));
                        }
                        _ => return Err(self.unexpected_character()),
                    }
                }
            }
            Some(b'{') => {
                self.position += 1;
                let mut entries = Vec::<(String, JsonValue)>::new();
                self.skip_whitespace();
                if self.peek() == Some(b'}') {
                    self.position += 1;
                    return Ok(JsonValue::Object(entries));
                }
                loop {
                    self.skip_whitespace();
                    if self.peek() != Some(b'"') {
                        return Err(self.unexpected_character());
                    }
                    let key = self.parse_string()?;
                    self.skip_whitespace();
                    if self.peek() != Some(b':') {
                        return Err(self.unexpected_character());
                    }
                    self.position += 1;
                    self.skip_whitespace();
                    let value = self.parse_value(depth + 1)?;
                    match entries.iter_mut().find(|(entry_key, _)| *entry_key == key) {
                        Some((_, entry_value)) => *entry_value = value,
                        None => entries.push((key, value)),
                    }
                    self.skip_whitespace();
                    match self.peek() {
                        Some(b',') => self.position += 1,
                        Some(b'}') => {
                            self.position += 1;
                            return Ok(JsonValue::Object(entries));
                        }
                        _ => return Err(self.unexpected_character()),
                    }
                }
            }
            _ => Err(self.unexpected_character()),
        }
    }

    /// Parses the literal `literal` at the current position, which then represents `value`.
    fn parse_literal(&mut self, literal: &str, value: JsonValue) -> Result<JsonValue, JsonError> {
        if self.json[self.position..].starts_with(literal) {
            self.position += literal.len();
            Ok(value)
        } else {
            Err(self.error(format!("expected `{literal}`")))
        }
    }

    /// Parses the number at the current position. Numbers with a fraction or an exponent are floats and all other
    /// numbers are integers.
    fn parse_number(&mut self) -> Result<JsonValue, JsonError> {
        let start = self.position;
        if self.peek() == Some(b'-') {
            self.position += 1;
        }
        match self.peek() {
            Some(b'0') => self.position += 1,
            Some(b'1'..=b'9') => self.skip_digits(),
            _ => return Err(self.error("expected a digit".to_string())),
        }
        let mut float = false;
        if self.peek() == Some(b'.') {
            float = true;
            self.position += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected a digit".to_string()));
            }
            self.skip_digits();
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            float = true;
            self.position += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.position += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected a digit".to_string()));
            }
            self.skip_digits();
        }
        let number = &self.json[start..self.position];
        let invalid_number = |description: &str| JsonError::Invalid {
            message: format!("number `{number}` at byte {start} {description}"),
        };
        if float {
            // Rust parses decimal numbers with correct rounding, so the result matches Python's `float`.
            let value = number.parse::<f64>().unwrap();
            if !value.is_finite() {
                return Err(invalid_number("is not finite"));
            }
            Ok(JsonValue::Float(value))
        } else {
            number
                .parse::<i64>()
                .map(JsonValue::Integer)
                .map_err(|_| invalid_number("does not fit in an `i64`"))
        }
    }

    /// Parses the string at the current position, which starts with `"`.
    fn parse_string(&mut self) -> Result<String, JsonError> {
        self.position += 1;
        let mut value = String::new();
        loop {
            let remaining = &self.json[self.position..];
            let Some(offset) = remaining.find(|character: char| matches!(character, '"' | '\\' | '\0'..='\u{1f}'))
            else {
                self.position = self.json.len();
                return Err(self.error("unterminated string".to_string()));
            };
            value.push_str(&remaining[..offset]);
            self.position += offset;
            match self.json.as_bytes()[self.position] {
                b'"' => {
                    self.position += 1;
                    return Ok(value);
                }
                b'\\' => {
                    self.position += 1;
                    let escaped = match self.peek() {
                        Some(b'"') => '"',
                        Some(b'\\') => '\\',
                        Some(b'/') => '/',
                        Some(b'b') => '\u{8}',
                        Some(b'f') => '\u{c}',
                        Some(b'n') => '\n',
                        Some(b'r') => '\r',
                        Some(b't') => '\t',
                        Some(b'u') => {
                            self.position += 1;
                            let high = self.parse_hexadecimal_code_unit()?;
                            let code_point = match high {
                                0xd800..=0xdbff if self.json[self.position..].starts_with("\\u") => {
                                    self.position += 2;
                                    let low = self.parse_hexadecimal_code_unit()?;
                                    if !(0xdc00..=0xdfff).contains(&low) {
                                        return Err(self.error("unpaired surrogate escape".to_string()));
                                    }
                                    0x10000 + ((high - 0xd800) << 10) + (low - 0xdc00)
                                }
                                0xd800..=0xdfff => return Err(self.error("unpaired surrogate escape".to_string())),
                                _ => high,
                            };
                            value.push(char::from_u32(code_point).unwrap());
                            continue;
                        }
                        _ => return Err(self.error("invalid escape sequence".to_string())),
                    };
                    value.push(escaped);
                    self.position += 1;
                }
                _ => return Err(self.error("unescaped control character in string".to_string())),
            }
        }
    }

    /// Parses the four hexadecimal digits of a `\u` escape at the current position.
    fn parse_hexadecimal_code_unit(&mut self) -> Result<u32, JsonError> {
        let digits = self.json.get(self.position..self.position + 4).unwrap_or_default();
        if digits.len() != 4 || !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(self.error("invalid unicode escape".to_string()));
        }
        self.position += 4;
        Ok(u32::from_str_radix(digits, 16).unwrap())
    }

    /// Advances past the ASCII digits at the current position.
    fn skip_digits(&mut self) {
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.position += 1;
        }
    }

    /// Advances past the JSON whitespace (i.e., spaces, tabs, line feeds, and carriage returns) at the current
    /// position.
    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.position += 1;
        }
    }

    /// Returns the byte at the current position, or [`None`] at the end of the input.
    fn peek(&self) -> Option<u8> {
        self.json.as_bytes().get(self.position).copied()
    }

    /// Returns the error for the unexpected character, or the unexpected end of the input, at the current position.
    fn unexpected_character(&self) -> JsonError {
        match self.json[self.position..].chars().next() {
            Some(character) => self.error(format!("unexpected character `{character}`")),
            None => self.error("unexpected end of input".to_string()),
        }
    }

    /// Returns an [`JsonError::Invalid`] error with `description` at the current position.
    fn error(&self, description: String) -> JsonError {
        JsonError::Invalid { message: format!("{description} at byte {}", self.position) }
    }
}

/// Appends `value` to `json` as a JSON string, spelled like Python's `json.dumps` with `ensure_ascii=False`.
fn write_string(value: &str, json: &mut String) {
    json.push('"');
    for character in value.chars() {
        match character {
            '"' => json.push_str("\\\""),
            '\\' => json.push_str("\\\\"),
            '\n' => json.push_str("\\n"),
            '\r' => json.push_str("\\r"),
            '\t' => json.push_str("\\t"),
            '\u{8}' => json.push_str("\\b"),
            '\u{c}' => json.push_str("\\f"),
            '\0'..='\u{1f}' => json.push_str(&format!("\\u{:04x}", character as u32)),
            _ => json.push(character),
        }
    }
    json.push('"');
}

/// Formats `value` like Python's `repr` of a float, which is also how `json.dumps` spells floats. The digits are the
/// shortest that round trip, the notation is positional when the decimal exponent is in `-4..16` (with a trailing
/// `.0` for integral values), and the notation is scientific otherwise, with a signed exponent of at least two digits.
fn format_float(value: f64) -> Result<String, JsonError> {
    if !value.is_finite() {
        return Err(JsonError::Invalid { message: format!("cannot serialize the non-finite float `{value}`") });
    }
    // `{:e}` produces the shortest round trip digits as `[-]d[.ddd]e[-]x`.
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let exponent = exponent.parse::<i32>().unwrap();
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(mantissa) => ("-", mantissa),
        None => ("", mantissa),
    };
    if !(-4..16).contains(&exponent) {
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        return Ok(format!("{sign}{mantissa}e{exponent_sign}{:02}", exponent.abs()));
    }
    let digits = mantissa.replace('.', "");
    let formatted = if exponent < 0 {
        format!("0.{}{digits}", "0".repeat(exponent.unsigned_abs() as usize - 1))
    } else {
        let integer_length = exponent as usize + 1;
        if digits.len() <= integer_length {
            format!("{digits}{}.0", "0".repeat(integer_length - digits.len()))
        } else {
            format!("{}.{}", &digits[..integer_length], &digits[integer_length..])
        }
    };
    Ok(format!("{sign}{formatted}"))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_json_value_parse() {
        assert_eq!(
            JsonValue::parse(" {\"b\": [1, 2.0, -3e2, true, false, null], \"a\": \"x\\u00e9\\ud83d\\ude00\\n\\/\"} "),
            Ok(JsonValue::Object(vec![
                (
                    "b".to_string(),
                    JsonValue::Array(vec![
                        JsonValue::Integer(1),
                        JsonValue::Float(2.0),
                        JsonValue::Float(-300.0),
                        JsonValue::Boolean(true),
                        JsonValue::Boolean(false),
                        JsonValue::Null,
                    ]),
                ),
                ("a".to_string(), JsonValue::String("xé😀\n/".to_string())),
            ])),
        );
        assert_eq!(JsonValue::parse("[]"), Ok(JsonValue::Array(Vec::new())));
        assert_eq!(JsonValue::parse("{}"), Ok(JsonValue::Object(Vec::new())));
        assert_eq!(JsonValue::parse("-0"), Ok(JsonValue::Integer(0)));
    }

    #[test]
    fn test_json_value_parse_repeated_keys() {
        // Like Python's `json.loads`, the first occurrence determines the position and the last one the value.
        assert_eq!(
            JsonValue::parse("{\"b\": 1, \"a\": 2, \"b\": 3}"),
            Ok(JsonValue::Object(vec![
                ("b".to_string(), JsonValue::Integer(3)),
                ("a".to_string(), JsonValue::Integer(2)),
            ])),
        );
    }

    #[test]
    fn test_json_value_parse_errors() {
        let cases = [
            ("", "unexpected end of input at byte 0"),
            ("nul", "expected `null` at byte 0"),
            ("NaN", "unexpected character `N` at byte 0"),
            ("[1,]", "unexpected character `]` at byte 3"),
            ("{\"a\" 1}", "unexpected character `1` at byte 5"),
            ("{1: 2}", "unexpected character `1` at byte 1"),
            ("[1] x", "unexpected character `x` at byte 4"),
            ("01", "unexpected character `1` at byte 1"),
            ("-", "expected a digit at byte 1"),
            ("1.", "expected a digit at byte 2"),
            ("1e+", "expected a digit at byte 3"),
            ("1e400", "number `1e400` at byte 0 is not finite"),
            ("9223372036854775808", "number `9223372036854775808` at byte 0 does not fit in an `i64`"),
            ("\"abc", "unterminated string at byte 4"),
            ("\"a\nb\"", "unescaped control character in string at byte 2"),
            ("\"\\x\"", "invalid escape sequence at byte 2"),
            ("\"\\u12\"", "invalid unicode escape at byte 3"),
            ("\"\\ud800\"", "unpaired surrogate escape at byte 7"),
            ("\"\\ud800\\u0041\"", "unpaired surrogate escape at byte 13"),
            ("\"\\udc00\"", "unpaired surrogate escape at byte 7"),
        ];
        for (json, expected_message) in cases {
            assert!(
                matches!(
                    JsonValue::parse(json),
                    Err(JsonError::Invalid { message }) if message == expected_message,
                ),
                "{json:?}",
            );
        }
        assert!(matches!(
            JsonValue::parse(&"[".repeat(129)),
            Err(JsonError::Invalid { message }) if message == "nesting exceeds 128 levels at byte 128",
        ));
    }

    #[test]
    fn test_json_value_get() {
        let value = JsonValue::parse("{\"name\": \"f\", \"args\": {}}").unwrap();
        assert_eq!(value.get("name"), Some(&JsonValue::String("f".to_string())));
        assert_eq!(value.get("args"), Some(&JsonValue::Object(Vec::new())));
        assert_eq!(value.get("other"), None);
        assert_eq!(JsonValue::Array(Vec::new()).get("name"), None);
    }

    #[test]
    fn test_json_value_to_json() {
        let value = JsonValue::parse("{\"b\": [1, 2.5, null, true], \"a\": {\"d\": false, \"c\": \"é\"}}").unwrap();
        assert_eq!(value.to_json(false), Ok("{\"b\":[1,2.5,null,true],\"a\":{\"d\":false,\"c\":\"é\"}}".to_string()));
        assert_eq!(value.to_json(true), Ok("{\"a\":{\"c\":\"é\",\"d\":false},\"b\":[1,2.5,null,true]}".to_string()));
        assert_eq!(
            JsonValue::String("a\"b\\c\né\u{1}\u{7f}\t\r\u{8}\u{c}/".to_string()).to_json(false),
            Ok("\"a\\\"b\\\\c\\né\\u0001\u{7f}\\t\\r\\b\\f/\"".to_string()),
        );
    }

    #[test]
    fn test_json_value_to_json_floats() {
        let cases = [
            (1.0, "1.0"),
            (0.5, "0.5"),
            (-2.5, "-2.5"),
            (-0.0, "-0.0"),
            (123.456, "123.456"),
            (0.1 + 0.2, "0.30000000000000004"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1e-7, "1e-07"),
            (1.5e-7, "1.5e-07"),
            (1e15, "1000000000000000.0"),
            (1234567890123456.7, "1234567890123456.8"),
            (1e16, "1e+16"),
            (1.5e300, "1.5e+300"),
            (5e-324, "5e-324"),
        ];
        for (value, expected) in cases {
            assert_eq!(JsonValue::Float(value).to_json(false), Ok(expected.to_string()), "{value:?}");
        }
        assert!(matches!(
            JsonValue::Array(vec![JsonValue::Float(f64::INFINITY)]).to_json(false),
            Err(JsonError::Invalid { message }) if message == "cannot serialize the non-finite float `inf`",
        ));
        assert!(matches!(
            JsonValue::Float(f64::NAN).to_json(false),
            Err(JsonError::Invalid { message }) if message == "cannot serialize the non-finite float `NaN`",
        ));
    }

    #[test]
    fn test_json_value_from() {
        let value = serde_json::json!({ "b": 1, "a": [1.5, null, true, "x"], "c": u64::MAX });
        assert_eq!(
            JsonValue::from(value),
            JsonValue::Object(vec![
                (
                    "a".to_string(),
                    JsonValue::Array(vec![
                        JsonValue::Float(1.5),
                        JsonValue::Null,
                        JsonValue::Boolean(true),
                        JsonValue::String("x".to_string()),
                    ]),
                ),
                ("b".to_string(), JsonValue::Integer(1)),
                ("c".to_string(), JsonValue::Float(18446744073709551615.0)),
            ]),
        );
    }
}
