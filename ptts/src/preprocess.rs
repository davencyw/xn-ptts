//! Text normalization applied before tokenization.
//!
//! The model is trained on ordinary written prose, so characters it never saw -- typographic
//! quotes, bullets, arrows, emoji -- are best removed rather than tokenized, and symbols that
//! are read aloud (`@`, `+`, `=`) are best spelled out in the target language. [`normalize_text`]
//! does both, driven by a [`Lang`].
//!
//! This is deliberately conservative: it does not expand numbers, dates or abbreviations, which
//! the model handles natively.

/// Spoken forms of the punctuation characters that are read aloud rather than dropped.
#[derive(Debug, Clone)]
pub struct SpecialChars {
    pub colon: &'static str,
    pub slash: &'static str,
    pub dash: &'static str,
    pub dot: &'static str,
    pub at: &'static str,
    pub plus: &'static str,
    pub equals: &'static str,
}

pub const SPECIAL_CHARS_EN: SpecialChars = SpecialChars {
    colon: "colon",
    slash: "slash",
    dash: "dash",
    dot: "dot",
    at: "at",
    plus: "plus",
    equals: "equals",
};

pub const SPECIAL_CHARS_FR: SpecialChars = SpecialChars {
    colon: "deux-points",
    slash: "slash",
    dash: "tiret",
    dot: "point",
    at: "arobaze",
    plus: "plus",
    equals: "égal",
};

pub const SPECIAL_CHARS_DE: SpecialChars = SpecialChars {
    colon: "Doppelpunkt",
    slash: "Slash",
    dash: "Bindestrich",
    dot: "Punkt",
    at: "ät",
    plus: "Plus",
    equals: "Gleich",
};

pub const SPECIAL_CHARS_ES: SpecialChars = SpecialChars {
    colon: "dos-puntos",
    slash: "slash",
    dash: "guion",
    dot: "punto",
    at: "arroba",
    plus: "mas",
    equals: "igual",
};

pub const SPECIAL_CHARS_PT: SpecialChars = SpecialChars {
    colon: "dois-pontos",
    slash: "slash",
    dash: "hifen",
    dot: "ponto",
    at: "arroba",
    plus: "mais",
    equals: "igual",
};

/// Language driving the spoken forms used by [`normalize_text`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    En,
    Fr,
    De,
    Es,
    Pt,
}

impl std::str::FromStr for Lang {
    type Err = xn::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "en" => Ok(Lang::En),
            "fr" => Ok(Lang::Fr),
            "de" => Ok(Lang::De),
            "es" => Ok(Lang::Es),
            "pt" => Ok(Lang::Pt),
            _ => xn::bail!("unsupported language code: {s}"),
        }
    }
}

impl Lang {
    pub fn special_chars(self) -> &'static SpecialChars {
        match self {
            Lang::En => &SPECIAL_CHARS_EN,
            Lang::Fr => &SPECIAL_CHARS_FR,
            Lang::De => &SPECIAL_CHARS_DE,
            Lang::Es => &SPECIAL_CHARS_ES,
            Lang::Pt => &SPECIAL_CHARS_PT,
        }
    }

    pub fn decimal_separator(self) -> &'static str {
        match self {
            Lang::En => "point",
            Lang::Fr => "virgule",
            Lang::De => "Komma",
            Lang::Es => "coma",
            Lang::Pt => "vírgula",
        }
    }

    pub fn underscore(self) -> &'static str {
        match self {
            Lang::En | Lang::Fr | Lang::Pt => "underscore",
            Lang::De => "Unterstrich",
            Lang::Es => "guion bajo",
        }
    }

    pub fn dollars(self) -> &'static str {
        match self {
            Lang::En | Lang::Fr => "dollars",
            Lang::De => "Dollar",
            Lang::Es | Lang::Pt => "dólares",
        }
    }

    pub fn dollars_singular(self) -> &'static str {
        match self {
            Lang::En | Lang::Fr => "dollar",
            Lang::De => "Dollar",
            Lang::Es | Lang::Pt => "dólar",
        }
    }

    pub fn euros(self) -> &'static str {
        match self {
            Lang::En | Lang::Fr | Lang::Es | Lang::Pt => "euros",
            Lang::De => "Euro",
        }
    }

    pub fn euros_singular(self) -> &'static str {
        match self {
            Lang::En | Lang::Fr | Lang::Es | Lang::Pt => "euro",
            Lang::De => "Euro",
        }
    }

    pub fn pounds(self) -> &'static str {
        match self {
            Lang::En => "pounds",
            Lang::Fr => "livres",
            Lang::De => "Pfund",
            Lang::Es | Lang::Pt => "libras",
        }
    }

    pub fn pounds_singular(self) -> &'static str {
        match self {
            Lang::En => "pound",
            Lang::Fr => "livre",
            Lang::De => "Pfund",
            Lang::Es | Lang::Pt => "libra",
        }
    }

    pub fn currency(&self, symbol: char) -> Option<&'static str> {
        match symbol {
            '$' => Some(self.dollars()),
            '€' => Some(self.euros()),
            '£' => Some(self.pounds()),
            _ => None,
        }
    }

    pub fn currency_singular(&self, symbol: char) -> Option<&'static str> {
        match symbol {
            '$' => Some(self.dollars_singular()),
            '€' => Some(self.euros_singular()),
            '£' => Some(self.pounds_singular()),
            _ => None,
        }
    }
}

fn is_emoji(c: char) -> bool {
    let c = c as u32;
    matches!(c,
        0x1F600..=0x1FAFF |
        0x2600..=0x27BF |
        // Flags (regional indicator symbols)
        0x1F1E6..=0x1F1FF
    )
}

/// Character sink that keeps the output free of the punctuation pile-ups the substitutions
/// below would otherwise produce: a `.` or `,` swallows any whitespace and punctuation
/// immediately before it, and runs of whitespace collapse to a single space.
struct StringAppender {
    buffer: Vec<char>,
}

impl StringAppender {
    fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    fn push(&mut self, c: char) {
        if c == '.' || c == ',' {
            while self.buffer.last().is_some_and(|l| l.is_whitespace() || l.is_ascii_punctuation())
            {
                self.buffer.pop();
            }
        }
        self.buffer.push(c);
    }

    fn push_str(&mut self, s: &str) {
        for c in s.chars() {
            self.push(c);
        }
    }

    fn into_string(mut self) -> String {
        self.pop_whitespace();
        self.buffer.into_iter().collect()
    }

    fn last_is_whitespace(&self) -> bool {
        self.buffer.last().is_some_and(|c| c.is_whitespace())
    }

    fn push_whitespace(&mut self) {
        if !self.last_is_whitespace() && !self.buffer.is_empty() {
            self.push(' ');
        }
    }

    fn pop_whitespace(&mut self) {
        while self.last_is_whitespace() {
            self.buffer.pop();
        }
    }
}

/// Rewrite `input` into the character set the model was trained on.
///
/// Typographic quotes, dashes, bullets, arrows and emoji are dropped or folded to their ASCII
/// equivalents; `@`, `+` and `=` are spelled out in `lang`; `;`, `:` and parentheses become
/// commas, which is how the model is asked to pause. Numbers, dates and abbreviations are left
/// alone -- the model reads those natively.
pub fn normalize_text(input: &str, lang: Lang) -> String {
    let mut res = StringAppender::new();
    for c in input.chars() {
        match c {
            '“' | '”' | '"' => res.push_whitespace(),
            '’' | '‘' => res.push('\''),
            '‐' | '‑' | '‒' | '―' => res.push('-'),
            // The two dashes below are not - (ascii 45) but similar unicode chars.
            '–' | '*' | '—' | '[' | ']' | '{' | '}' => res.push_whitespace(),
            '•' | '‣' | '◦' | '·' | '→' | '←' | '↑' | '↓' | '➡' | '➜' => {
                res.push_whitespace();
            }
            '…' => res.push('.'),
            '@' => {
                res.push_whitespace();
                res.push_str(lang.special_chars().at);
                res.push_whitespace();
            }
            '+' => {
                res.push_whitespace();
                res.push_str(lang.special_chars().plus);
                res.push_whitespace();
            }
            '=' => {
                res.push_whitespace();
                res.push_str(lang.special_chars().equals);
                res.push_whitespace();
            }
            ';' | ':' | '(' | ')' => {
                res.pop_whitespace();
                res.push(',');
                res.push_whitespace();
            }
            c => {
                if is_emoji(c) || c.is_control() || c.is_whitespace() {
                    res.push_whitespace();
                } else {
                    res.push(c)
                }
            }
        }
    }
    res.into_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_text_cases() {
        let cases: &[(&str, &str)] = &[
            ("Hello, world!", "Hello, world!"),
            ("", ""),
            ("“hello” world it's", "hello world it's"),
            ("a‐b‑c‒d―e", "a-b-c-d-e"),
            ("a–b—c", "a b c"),
            ("foo (bar) [baz] {qux} *quux*", "foo, bar, baz qux quux"),
            ("• ‣ ◦ · a→b←c↑d↓e ➡ ➜", "a b c d e"),
            ("wait… a…b", "wait. a.b"),
            ("user@host @home", "user at host at home"),
            ("café résumé 日本語", "café résumé 日本語"),
            ("hello 😀 flag 🇫🇷 sun ☀", "hello flag sun"),
            // ';', ':' and '(' / ')' all collapse to ", " (comma + single space).
            ("a;b:c", "a, b, c"),
            ("time: 10:30", "time, 10, 30"),
            ("; leading", ", leading"),
            ("hello (world)", "hello, world,"),
            // Surrounding whitespace is absorbed into the comma replacement.
            ("foo ; bar  :  baz", "foo, bar, baz"),
            // Runs of ASCII and non-ASCII whitespace collapse to a single space,
            // and trailing whitespace is stripped.
            ("a   b\t\tc\n\nd", "a b c d"),
            ("hello   ", "hello"),
            ("a • b • c", "a b c"),
            ("“Hello”; please email user@host (now)… 🚀", "Hello, please email user at host, now."),
            (
                "Numbers: one, two, three, four, five. Special items: at sign, hash, dollar, percent.",
                "Numbers, one, two, three, four, five. Special items, at sign, hash, dollar, percent.",
            ),
            (
                "The conference will be held on Tuesday, March 15th at 3:30 PM.",
                "The conference will be held on Tuesday, March 15th at 3, 30 PM.",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(&normalize_text(input, Lang::En), expected, "input: {input:?}");
        }
    }

    #[test]
    fn spoken_symbols_follow_the_language() {
        assert_eq!(normalize_text("a@b", Lang::En), "a at b");
        assert_eq!(normalize_text("a@b", Lang::Fr), "a arobaze b");
        assert_eq!(normalize_text("a@b", Lang::De), "a ät b");
        assert_eq!(normalize_text("1+1=2", Lang::Es), "1 mas 1 igual 2");
        assert_eq!(normalize_text("1+1=2", Lang::Pt), "1 mais 1 igual 2");
    }

    #[test]
    fn lang_round_trips_through_str() {
        use std::str::FromStr;
        for (s, lang) in [
            ("en", Lang::En),
            ("FR", Lang::Fr),
            ("de", Lang::De),
            ("es", Lang::Es),
            ("pt", Lang::Pt),
        ] {
            assert_eq!(Lang::from_str(s).unwrap(), lang);
        }
        assert!(Lang::from_str("klingon").is_err());
    }
}
