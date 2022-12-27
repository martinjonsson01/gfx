use cgmath::{Deg, InnerSpace, Quaternion, Rotation3, Vector3, Zero};
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

    const NUM_INSTANCES_PER_ROW: u32 = 10;
    const SPACE_BETWEEN: f32 = 10.0;
    let transforms = (0..NUM_INSTANCES_PER_ROW)
        .flat_map(|z| {
            (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                let position = Vector3 { x, y: 0.0, z };

                let rotation = if position.is_zero() {
                    // This is needed so an object at (0, 0, 0) won't get scaled to zero
                    // as Quaternions can effect scale if they're not created correctly.
                    Quaternion::from_axis_angle(Vector3::unit_z(), Deg(0.0))
                } else {
                    Quaternion::from_axis_angle(position.normalize(), Deg(45.0))
                };

                Transform { position, rotation }
            })
        })
        .collect::<Vec<_>>();

    gfx.create_objects(model, transforms)?;

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
