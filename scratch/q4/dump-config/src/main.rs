fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = ptts::tts_model::TTSConfig::v202601(0.7);
    println!("{}", serde_json::to_string_pretty(&cfg)?);
    Ok(())
}
