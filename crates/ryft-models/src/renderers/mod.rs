//! Renderers, which turn structured conversations into the token sequences that chat models were trained on and turn
//! sampled tokens back into messages.
//!
//! A [`Conversation`] is a list of [`Message`]s together with the [`Tool`]s that the model may call and an optional
//! [`ReasoningEffort`]. Each message has a [`Role`] and may carry [`ContentPart`]s (i.e., text and image
//! placeholders), reasoning text, and [`ToolCall`]s. These types are a small vocabulary that is independent of any
//! particular model, and a [`Renderer`] implements the format of one model family on top of them, using a
//! [`Tokenizer`](crate::tokenizers::Tokenizer) for the text in the format.
//!
//! [`Renderer::render`] produces a [`RenderedConversation`], which pairs the tokens of a conversation with a
//! supervision mask that marks the tokens that a model is trained to predict (i.e., those of the assistant turns) and
//! with the token ranges that image placeholders occupy, which a vision encoder later fills. Sampling an assistant
//! turn stops at one of the [`Renderer::stop_tokens`], and [`Renderer::parse`] converts the sampled tokens back into
//! an assistant [`Message`], so that rendering that message and parsing its tokens round trips.
//!
//! [`Inkling2Renderer`] implements the format of Inkling 2 models. Tool arguments and tool parameter schemas are
//! [`JsonValue`]s, which preserve the order of object keys because the spelling of JSON in rendered conversations
//! depends on it.
//!
//! # Release Parity Tests
//!
//! The unit tests of this module use a small synthetic tokenizer. An additional ignored test checks
//! [`Inkling2Renderer`] against token sequences and supervision masks that the reference chat template of the
//! Inkling 2 release produced. It reads `tokenizer.json` from the directory named by the
//! `RYFT_MODELS_INKLING2_RELEASE_DIRECTORY` environment variable and runs with `cargo test --release -- --ignored`.

use std::fmt::Display;
use std::ops::Range;

use thiserror::Error;

use crate::json::{JsonError, JsonValue};
use crate::tokenizers::{TokenId, TokenizerError};

pub mod inkling2;

pub use inkling2::Inkling2Renderer;

// TODO(eaplatanios): Review this module.

/// Error produced when constructing a renderer, rendering a conversation, or parsing sampled tokens.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum RendererError {
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),

    #[error("the tokenizer has no `{token}` token")]
    MissingSpecialToken { token: String },

    #[error("invalid conversation: {message}")]
    InvalidConversation { message: String },

    #[error("invalid rendered conversation: {message}")]
    InvalidRenderedConversation { message: String },

    #[error(transparent)]
    Json(#[from] JsonError),

    #[error("malformed response: {message}")]
    MalformedResponse { message: String },
}

/// Format of the conversations of one model family, which converts [`Conversation`]s to tokens and sampled tokens
/// back to [`Message`]s.
pub trait Renderer {
    /// Renders `conversation` into tokens, a supervision mask, and image placeholder spans.
    fn render(&self, conversation: &Conversation) -> Result<RenderedConversation, RendererError>;

    /// Returns the token identifiers at which sampling of an assistant turn stops.
    fn stop_tokens(&self) -> &[TokenId];

    /// Parses the tokens of one sampled assistant turn, with or without its trailing stop token, into a message.
    fn parse(&self, tokens: &[TokenId]) -> Result<Message, RendererError>;
}

/// Author of a [`Message`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// Instructions that configure the behavior of the model.
    System,

    /// Input of the user.
    User,

    /// Output of the model.
    Assistant,

    /// Result of a tool call of the model.
    Tool,
}

impl Display for Role {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => write!(formatter, "system"),
            Self::User => write!(formatter, "user"),
            Self::Assistant => write!(formatter, "assistant"),
            Self::Tool => write!(formatter, "tool"),
        }
    }
}

/// Part of the content of a [`Message`].
#[derive(Clone, Debug, PartialEq)]
pub enum ContentPart {
    /// Text.
    Text(String),

    /// Image, which occupies placeholder tokens that the embeddings of a vision encoder later replace.
    Image {
        /// Number of placeholder tokens that the image occupies.
        token_count: usize,
    },
}

/// Call of a [`Tool`] by the model.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    /// Optional identifier that [`Role::Tool`] messages use to refer to this call.
    id: Option<String>,

    /// Name of the called function.
    name: String,

    /// Arguments of the call, which form a JSON object.
    arguments: JsonValue,
}

impl ToolCall {
    /// Creates a [`ToolCall`] of function `name` with `arguments`, which must be a [`JsonValue::Object`].
    pub fn new(name: impl Into<String>, arguments: JsonValue) -> Result<Self, RendererError> {
        let name = name.into();
        if !matches!(arguments, JsonValue::Object(_)) {
            return Err(RendererError::InvalidConversation {
                message: format!("the arguments of the tool call of `{name}` must be a JSON object"),
            });
        }
        Ok(Self { id: None, name, arguments })
    }

    /// Returns this [`ToolCall`] with the identifier `id`.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Returns the identifier of this call, if it has one.
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Returns the name of the called function.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the arguments of this call, which form a [`JsonValue::Object`].
    pub fn arguments(&self) -> &JsonValue {
        &self.arguments
    }
}

/// Tool that a [`Conversation`] declares to the model, which can call it with [`ToolCall`]s.
#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    /// Name of the function.
    name: String,

    /// Description of the function for the model.
    description: String,

    /// JSON schema of the arguments of the function.
    parameters: JsonValue,

    /// Type of the tool.
    r#type: String,
}

impl Tool {
    /// Creates a [`Tool`] for function `name` with an empty description, an empty object as its parameter schema, and
    /// the type `function`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            parameters: JsonValue::Object(Vec::new()),
            r#type: "function".to_string(),
        }
    }

    /// Returns this [`Tool`] with the description `description`.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Returns this [`Tool`] with the JSON schema `parameters` for the arguments of its function.
    pub fn with_parameters(mut self, parameters: JsonValue) -> Self {
        self.parameters = parameters;
        self
    }

    /// Returns this [`Tool`] with the type `type`.
    pub fn with_type(mut self, r#type: impl Into<String>) -> Self {
        self.r#type = r#type.into();
        self
    }

    /// Returns the name of the function of this tool.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the description of this tool.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns the JSON schema of the arguments of the function of this tool.
    pub fn parameters(&self) -> &JsonValue {
        &self.parameters
    }

    /// Returns the type of this tool.
    pub fn r#type(&self) -> &str {
        &self.r#type
    }
}

/// Message of a [`Conversation`].
#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    /// Author of this message.
    role: Role,

    /// Optional name of the author (e.g., the name of the function that produced a [`Role::Tool`] message).
    name: Option<String>,

    /// Content of this message, in order.
    content: Vec<ContentPart>,

    /// Optional reasoning of an assistant message, which precedes its content.
    reasoning: Option<String>,

    /// Tool calls of an assistant message, which follow its content.
    tool_calls: Vec<ToolCall>,

    /// Identifier of the [`ToolCall`] whose result a [`Role::Tool`] message holds.
    tool_call_id: Option<String>,
}

impl Message {
    /// Creates an empty [`Message`] with `role`.
    pub fn new(role: Role) -> Self {
        Self { role, name: None, content: Vec::new(), reasoning: None, tool_calls: Vec::new(), tool_call_id: None }
    }

    /// Creates a [`Message`] with `role` whose content is one [`ContentPart::Text`] part with `text`.
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self::new(role).with_content_part(ContentPart::Text(text.into()))
    }

    /// Returns this [`Message`] with the author name `name`.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns this [`Message`] with `part` appended to its content.
    pub fn with_content_part(mut self, part: ContentPart) -> Self {
        self.content.push(part);
        self
    }

    /// Returns this [`Message`] with the reasoning `reasoning`.
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }

    /// Returns this [`Message`] with `tool_call` appended to its tool calls.
    pub fn with_tool_call(mut self, tool_call: ToolCall) -> Self {
        self.tool_calls.push(tool_call);
        self
    }

    /// Returns this [`Message`] with the tool call identifier `tool_call_id`.
    pub fn with_tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(tool_call_id.into());
        self
    }

    /// Returns the author of this message.
    pub fn role(&self) -> Role {
        self.role
    }

    /// Returns the name of the author of this message, if it has one.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the content of this message.
    pub fn content(&self) -> &[ContentPart] {
        &self.content
    }

    /// Returns the reasoning of this message, if it has any.
    pub fn reasoning(&self) -> Option<&str> {
        self.reasoning.as_deref()
    }

    /// Returns the tool calls of this message.
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    /// Returns the identifier of the [`ToolCall`] whose result this message holds, if it has one.
    pub fn tool_call_id(&self) -> Option<&str> {
        self.tool_call_id.as_deref()
    }
}

/// Amount of reasoning that a model should do before it answers.
#[derive(Clone, Debug, PartialEq)]
pub enum ReasoningEffort {
    /// Named level (e.g., `low` or `high`), whose meaning depends on the [`Renderer`].
    Named(String),

    /// Numeric level, whose valid range depends on the [`Renderer`].
    Value(f64),
}

/// Conversation that a [`Renderer`] renders, consisting of messages, declared tools, and a reasoning effort.
#[derive(Clone, Debug, PartialEq)]
pub struct Conversation {
    /// Messages of this conversation, in order.
    messages: Vec<Message>,

    /// Tools that this conversation declares to the model.
    tools: Vec<Tool>,

    /// Optional reasoning effort, where [`None`] selects the default effort of the [`Renderer`].
    reasoning_effort: Option<ReasoningEffort>,
}

impl Conversation {
    /// Creates a [`Conversation`] with `messages`, no tools, and the default reasoning effort.
    pub fn new(messages: Vec<Message>) -> Self {
        Self { messages, tools: Vec::new(), reasoning_effort: None }
    }

    /// Returns this [`Conversation`] with `tool` appended to its declared tools.
    pub fn with_tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Returns this [`Conversation`] with the reasoning effort `reasoning_effort`.
    pub fn with_reasoning_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(reasoning_effort);
        self
    }

    /// Returns the messages of this conversation.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Returns the tools that this conversation declares.
    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    /// Returns the reasoning effort of this conversation, where [`None`] selects the default of the [`Renderer`].
    pub fn reasoning_effort(&self) -> Option<&ReasoningEffort> {
        self.reasoning_effort.as_ref()
    }
}

/// Output of [`Renderer::render`], which holds the tokens of a [`Conversation`] together with their supervision mask
/// and the token ranges of image placeholders.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedConversation {
    /// Tokens of the conversation.
    tokens: Vec<TokenId>,

    /// Supervision mask with one entry per token, which is `true` for the tokens that a model is trained to predict.
    supervision: Vec<bool>,

    /// Ranges of [`Self::tokens`] that image placeholders occupy, in order.
    image_spans: Vec<Range<usize>>,
}

impl RenderedConversation {
    /// Creates a [`RenderedConversation`]. `supervision` must have one entry per token, and `image_spans` must be
    /// ordered, non-overlapping ranges of token positions.
    pub fn new(
        tokens: Vec<TokenId>,
        supervision: Vec<bool>,
        image_spans: Vec<Range<usize>>,
    ) -> Result<Self, RendererError> {
        if supervision.len() != tokens.len() {
            return Err(RendererError::InvalidRenderedConversation {
                message: format!(
                    "the supervision mask has {} entries but there are {} tokens",
                    supervision.len(),
                    tokens.len(),
                ),
            });
        }
        let mut end = 0;
        for span in &image_spans {
            if span.start < end || span.start > span.end || span.end > tokens.len() {
                return Err(RendererError::InvalidRenderedConversation {
                    message: format!(
                        "image span {span:?} is not an ordered and non-overlapping range of the {} tokens",
                        tokens.len(),
                    ),
                });
            }
            end = span.end;
        }
        Ok(Self { tokens, supervision, image_spans })
    }

    /// Returns the tokens of the conversation.
    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }

    /// Returns the supervision mask, which has one entry per token and is `true` for the tokens that a model is
    /// trained to predict.
    pub fn supervision(&self) -> &[bool] {
        &self.supervision
    }

    /// Returns the token ranges that image placeholders occupy, in order.
    pub fn image_spans(&self) -> &[Range<usize>] {
        &self.image_spans
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::tokenizers::{SpecialTokenHandling, Tokenizer};

    use super::*;

    /// [`Tokenizer`] with one token per byte, whose identifier is the value of that byte, followed by special tokens.
    pub(super) struct TestTokenizer {
        /// Strings of the tokens indexed by identifier, where bytes outside ASCII are spelled like `<0x80>`.
        tokens: Vec<String>,
    }

    impl TestTokenizer {
        /// Creates a [`TestTokenizer`] with `special_tokens`, whose identifiers are 256 onward in order.
        pub(super) fn new(special_tokens: &[&str]) -> Self {
            let tokens = (0..=u8::MAX)
                .map(|byte| if byte.is_ascii() { char::from(byte).to_string() } else { format!("<0x{byte:02X}>") })
                .chain(special_tokens.iter().map(|token| token.to_string()))
                .collect();
            Self { tokens }
        }
    }

    impl Tokenizer for TestTokenizer {
        fn encode(&self, text: &str, special_tokens: SpecialTokenHandling) -> Result<Vec<TokenId>, TokenizerError> {
            let bytes = text.as_bytes();
            let mut tokens = Vec::new();
            let mut position = 0;
            while position < bytes.len() {
                let special_token = match special_tokens {
                    SpecialTokenHandling::Parse => {
                        (256..self.tokens.len()).find(|id| bytes[position..].starts_with(self.tokens[*id].as_bytes()))
                    }
                    SpecialTokenHandling::Literal => None,
                };
                match special_token {
                    Some(id) => {
                        tokens.push(id as TokenId);
                        position += self.tokens[id].len();
                    }
                    None => {
                        tokens.push(TokenId::from(bytes[position]));
                        position += 1;
                    }
                }
            }
            Ok(tokens)
        }

        fn decode(&self, tokens: &[TokenId], skip_special_tokens: bool) -> Result<String, TokenizerError> {
            let mut bytes = Vec::new();
            for id in tokens {
                match *id {
                    0..256 => bytes.push(*id as u8),
                    _ if skip_special_tokens => {}
                    _ => match self.tokens.get(*id as usize) {
                        Some(token) => bytes.extend_from_slice(token.as_bytes()),
                        None => {
                            return Err(TokenizerError::Decoding { message: format!("unknown token identifier {id}") });
                        }
                    },
                }
            }
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }

        fn vocabulary_size(&self) -> usize {
            self.tokens.len()
        }

        fn token_id(&self, token: &str) -> Option<TokenId> {
            self.tokens.iter().position(|candidate| candidate == token).map(|id| id as TokenId)
        }

        fn token(&self, id: TokenId) -> Option<&str> {
            self.tokens.get(id as usize).map(String::as_str)
        }
    }

    #[test]
    fn test_role_display() {
        assert_eq!(Role::System.to_string(), "system");
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Assistant.to_string(), "assistant");
        assert_eq!(Role::Tool.to_string(), "tool");
    }

    #[test]
    fn test_tool_call_new() {
        let arguments = JsonValue::Object(vec![("city".to_string(), JsonValue::String("Paris".to_string()))]);
        let tool_call = ToolCall::new("get_weather", arguments.clone()).unwrap();
        assert_eq!(tool_call.id(), None);
        assert_eq!(tool_call.name(), "get_weather");
        assert_eq!(tool_call.arguments(), &arguments);

        assert!(matches!(
            ToolCall::new("get_weather", JsonValue::Array(Vec::new())),
            Err(RendererError::InvalidConversation { message })
                if message == "the arguments of the tool call of `get_weather` must be a JSON object",
        ));
    }

    #[test]
    fn test_tool_call_with_id() {
        let tool_call = ToolCall::new("f", JsonValue::Object(Vec::new())).unwrap().with_id("call_1");
        assert_eq!(tool_call.id(), Some("call_1"));
    }

    #[test]
    fn test_tool_call_id() {
        let tool_call = ToolCall::new("f", JsonValue::Object(Vec::new())).unwrap();
        assert_eq!(tool_call.id(), None);
        assert_eq!(tool_call.with_id("call_2").id(), Some("call_2"));
    }

    #[test]
    fn test_tool_call_name() {
        assert_eq!(ToolCall::new("f", JsonValue::Object(Vec::new())).unwrap().name(), "f");
    }

    #[test]
    fn test_tool_call_arguments() {
        let arguments = JsonValue::Object(vec![("x".to_string(), JsonValue::Integer(1))]);
        assert_eq!(ToolCall::new("f", arguments.clone()).unwrap().arguments(), &arguments);
    }

    #[test]
    fn test_tool_new() {
        let tool = Tool::new("search");
        assert_eq!(tool.name(), "search");
        assert_eq!(tool.description(), "");
        assert_eq!(tool.parameters(), &JsonValue::Object(Vec::new()));
        assert_eq!(tool.r#type(), "function");
    }

    #[test]
    fn test_tool_with_description() {
        assert_eq!(Tool::new("search").with_description("Searches the web.").description(), "Searches the web.");
    }

    #[test]
    fn test_tool_with_parameters() {
        let parameters = JsonValue::Object(vec![("type".to_string(), JsonValue::String("object".to_string()))]);
        assert_eq!(Tool::new("search").with_parameters(parameters.clone()).parameters(), &parameters);
    }

    #[test]
    fn test_tool_with_type() {
        assert_eq!(Tool::new("search").with_type("custom").r#type(), "custom");
    }

    #[test]
    fn test_tool_name() {
        assert_eq!(Tool::new("search").name(), "search");
    }

    #[test]
    fn test_tool_description() {
        assert_eq!(Tool::new("search").description(), "");
    }

    #[test]
    fn test_tool_parameters() {
        assert_eq!(Tool::new("search").parameters(), &JsonValue::Object(Vec::new()));
    }

    #[test]
    fn test_tool_type() {
        assert_eq!(Tool::new("search").r#type(), "function");
    }

    #[test]
    fn test_message_new() {
        let message = Message::new(Role::User);
        assert_eq!(message.role(), Role::User);
        assert_eq!(message.name(), None);
        assert_eq!(message.content(), &[]);
        assert_eq!(message.reasoning(), None);
        assert_eq!(message.tool_calls(), &[]);
        assert_eq!(message.tool_call_id(), None);
    }

    #[test]
    fn test_message_text() {
        let message = Message::text(Role::Assistant, "Hello!");
        assert_eq!(message.role(), Role::Assistant);
        assert_eq!(message.content(), &[ContentPart::Text("Hello!".to_string())]);
    }

    #[test]
    fn test_message_with_name() {
        assert_eq!(Message::new(Role::Tool).with_name("search").name(), Some("search"));
    }

    #[test]
    fn test_message_with_content_part() {
        let message = Message::text(Role::User, "Look.").with_content_part(ContentPart::Image { token_count: 4 });
        let expected_content = [ContentPart::Text("Look.".to_string()), ContentPart::Image { token_count: 4 }];
        assert_eq!(message.content(), &expected_content);
    }

    #[test]
    fn test_message_with_reasoning() {
        assert_eq!(Message::new(Role::Assistant).with_reasoning("Hmm.").reasoning(), Some("Hmm."));
    }

    #[test]
    fn test_message_with_tool_call() {
        let first = ToolCall::new("f", JsonValue::Object(Vec::new())).unwrap();
        let second = ToolCall::new("g", JsonValue::Object(Vec::new())).unwrap();
        let message = Message::new(Role::Assistant).with_tool_call(first.clone()).with_tool_call(second.clone());
        assert_eq!(message.tool_calls(), &[first, second]);
    }

    #[test]
    fn test_message_with_tool_call_id() {
        assert_eq!(Message::new(Role::Tool).with_tool_call_id("call_1").tool_call_id(), Some("call_1"));
    }

    #[test]
    fn test_message_role() {
        assert_eq!(Message::new(Role::System).role(), Role::System);
    }

    #[test]
    fn test_message_name() {
        assert_eq!(Message::new(Role::Assistant).name(), None);
        assert_eq!(Message::new(Role::Assistant).with_name("helper").name(), Some("helper"));
    }

    #[test]
    fn test_message_content() {
        assert_eq!(Message::text(Role::User, "Hi").content(), &[ContentPart::Text("Hi".to_string())]);
    }

    #[test]
    fn test_message_reasoning() {
        assert_eq!(Message::new(Role::Assistant).reasoning(), None);
        assert_eq!(Message::new(Role::Assistant).with_reasoning("").reasoning(), Some(""));
    }

    #[test]
    fn test_message_tool_calls() {
        assert_eq!(Message::new(Role::Assistant).tool_calls(), &[]);
    }

    #[test]
    fn test_message_tool_call_id() {
        assert_eq!(Message::new(Role::Tool).tool_call_id(), None);
    }

    #[test]
    fn test_conversation_new() {
        let conversation = Conversation::new(vec![Message::text(Role::User, "Hi")]);
        assert_eq!(conversation.messages(), &[Message::text(Role::User, "Hi")]);
        assert_eq!(conversation.tools(), &[]);
        assert_eq!(conversation.reasoning_effort(), None);
    }

    #[test]
    fn test_conversation_with_tool() {
        let conversation = Conversation::new(Vec::new()).with_tool(Tool::new("f")).with_tool(Tool::new("g"));
        assert_eq!(conversation.tools(), &[Tool::new("f"), Tool::new("g")]);
    }

    #[test]
    fn test_conversation_with_reasoning_effort() {
        let conversation = Conversation::new(Vec::new()).with_reasoning_effort(ReasoningEffort::Value(0.5));
        assert_eq!(conversation.reasoning_effort(), Some(&ReasoningEffort::Value(0.5)));
    }

    #[test]
    fn test_conversation_messages() {
        assert_eq!(Conversation::new(Vec::new()).messages(), &[]);
    }

    #[test]
    fn test_conversation_tools() {
        assert_eq!(Conversation::new(Vec::new()).with_tool(Tool::new("f")).tools(), &[Tool::new("f")]);
    }

    #[test]
    fn test_conversation_reasoning_effort() {
        let effort = ReasoningEffort::Named("low".to_string());
        let conversation = Conversation::new(Vec::new()).with_reasoning_effort(effort.clone());
        assert_eq!(conversation.reasoning_effort(), Some(&effort));
    }

    #[test]
    fn test_rendered_conversation_new() {
        let rendered = RenderedConversation::new(vec![1, 2, 3, 4], vec![false, false, true, true], vec![1..2, 2..2]);
        let rendered = rendered.unwrap();
        assert_eq!(rendered.tokens(), &[1, 2, 3, 4]);
        assert_eq!(rendered.supervision(), &[false, false, true, true]);
        assert_eq!(rendered.image_spans(), &[1..2, 2..2]);

        assert!(matches!(
            RenderedConversation::new(vec![1, 2], vec![false], Vec::new()),
            Err(RendererError::InvalidRenderedConversation { message })
                if message == "the supervision mask has 1 entries but there are 2 tokens",
        ));
        assert!(matches!(
            RenderedConversation::new(vec![1, 2], vec![false, false], vec![0..0, 1..3]),
            Err(RendererError::InvalidRenderedConversation { message })
                if message == "image span 1..3 is not an ordered and non-overlapping range of the 2 tokens",
        ));
        assert!(matches!(
            RenderedConversation::new(vec![1, 2], vec![false, false], vec![0..2, 1..2]),
            Err(RendererError::InvalidRenderedConversation { message })
                if message == "image span 1..2 is not an ordered and non-overlapping range of the 2 tokens",
        ));
    }

    #[test]
    fn test_rendered_conversation_tokens() {
        assert_eq!(RenderedConversation::new(vec![7], vec![true], Vec::new()).unwrap().tokens(), &[7]);
    }

    #[test]
    fn test_rendered_conversation_supervision() {
        assert_eq!(RenderedConversation::new(vec![7], vec![true], Vec::new()).unwrap().supervision(), &[true]);
    }

    #[test]
    fn test_rendered_conversation_image_spans() {
        let rendered = RenderedConversation::new(vec![7], vec![false], vec![0..0, 0..1]).unwrap();
        assert_eq!(rendered.image_spans(), &[0..0, 0..1]);
    }
}
