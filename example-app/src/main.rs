use cgmath::{Quaternion, Vector3, Zero};
use color_eyre::eyre::Result;
use color_eyre::Report;
use recs_gfx::{GraphicsEngine, Transform};
use tracing::{info_span, instrument};

#[instrument]
fn main() -> Result<(), Report> {
    install_tracing()?;

    color_eyre::install()?;

    info_span!("gfx").in_scope(|| pollster::block_on(async_main()))?;

    Ok(())
}

async fn async_main() -> Result<(), Report> {
    let mut gfx = GraphicsEngine::new().await?;

    let model = gfx.load_model("cube.obj").await?;
    let transform = Transform {
        position: Vector3::zero(),
        rotation: Quaternion::zero(),
    };
    gfx.create_object(model, transform)?;

    gfx.run().await?;
    Ok(())
}

fn install_tracing() -> Result<(), Report> {
    use tracing_error::ErrorLayer;
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::{fmt, EnvFilter};

    let fmt_layer = fmt::layer().with_target(false);
    let filter_layer = EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info"))?;

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .with(ErrorLayer::default())
        .init();

    Ok(())
}
