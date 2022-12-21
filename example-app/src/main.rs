use recs_gfx::{run, EngineError};

fn main() -> Result<(), EngineError> {
    pollster::block_on(run())?;
    Ok(())
}
