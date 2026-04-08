/// 24-bit RGB color.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Color {
    /// Red channel.
    pub red: u8,

    /// Green channel.
    pub green: u8,

    /// Blue channel.
    pub blue: u8,
}

impl Color {
    /// Pure black [`Color`] (i.e., `#000000`).
    pub const BLACK: Self = Self { red: 0, green: 0, blue: 0 };

    /// Pure white [`Color`] (i.e., `#FFFFFF`).
    pub const WHITE: Self = Self { red: 255, green: 255, blue: 255 };

    /// Creates a new [`Color`] from the provided red, green, and blue channel values.
    pub const fn new(red: u8, green: u8, blue: u8) -> Self {
        Self { red, green, blue }
    }

    /// Returns the perceived [luminance](https://en.wikipedia.org/wiki/Relative_luminance) of this [`Color`] on
    /// a `0.0..=255.0` scale using the [ITU-R BT.601](https://en.wikipedia.org/wiki/Rec._601) luma coefficients.
    pub fn luminance(self) -> f32 {
        f32::from(self.red) * 0.299 + f32::from(self.green) * 0.587 + f32::from(self.blue) * 0.114
    }

    /// Returns [`Color::BLACK`] or [`Color::WHITE`] depending on which provides better contrast when this [`Color`]
    /// is used as a background color and the returned color is used as the foreground color.
    pub fn foreground_color(self) -> Self {
        if self.luminance() > 186.0 { Self::BLACK } else { Self::WHITE }
    }

    /// Wraps `text` in ANSI 24-bit true-color escape sequences that set the given `foreground`
    /// and `background` colors, followed by a reset.
    pub fn colored_text(text: &str, foreground: Self, background: Self) -> String {
        format!(
            "\u{1b}[38;2;{};{};{}m\u{1b}[48;2;{};{};{}m{text}\u{1b}[0m",
            foreground.red, foreground.green, foreground.blue, background.red, background.green, background.blue,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color() {
        assert_eq!(Color::BLACK.luminance(), 0.0);
        assert!(Color::WHITE.luminance() > 186.0);
        assert_eq!(Color::new(57, 59, 121).foreground_color(), Color::WHITE);
        assert_eq!(Color::new(231, 203, 148).foreground_color(), Color::BLACK);
        assert_eq!(
            Color::colored_text("hello", Color::WHITE, Color::BLACK),
            "\u{1b}[38;2;255;255;255m\u{1b}[48;2;0;0;0mhello\u{1b}[0m",
        );
    }
}
