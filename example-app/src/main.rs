use cgmath::{Deg, InnerSpace, Quaternion, Rotation3, Vector3, Zero};
use color_eyre::eyre::Result;
use color_eyre::Report;
use rand::Rng;
use recs_gfx::{EngineResult, GraphicsEngine, Object, SimulationBuffer, Transform};
use std::path::Path;
use std::thread;
use std::time::Duration;
use tracing::{info, instrument};

#[instrument]
fn main() -> Result<(), Report> {
    install_tracing()?;

    color_eyre::install()?;

    fn init_gfx(gfx: &mut GraphicsEngine<'_>) -> EngineResult<()> {
        let model = gfx.load_model(Path::new("cube.obj"))?;

        let transforms = create_transforms();
        gfx.create_objects(model, transforms)?;

        Ok(())
    }

    fn simulate(queue: &SimulationBuffer<Vec<Object>>) {
        info!("testing {queue:?}");
        thread::sleep(Duration::from_secs(1));
    }

    recs_gfx::start(init_gfx, simulate)?;

    Ok(())
}

fn create_transforms() -> Vec<Transform> {
    const NUM_INSTANCES_PER_ROW: u32 = 10;
    const SPACE_BETWEEN: f32 = 10.0;
    (0..NUM_INSTANCES_PER_ROW)
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

                let mut random = rand::thread_rng();
                let scale = [
                    random.gen_range(0.1..1.0),
                    random.gen_range(0.1..1.0),
                    random.gen_range(0.1..1.0),
                ]
                .into();

                Transform {
                    position,
                    rotation,
                    scale,
                }
            })
        })
        .collect()
}

fn install_tracing() -> Result<(), Report> {
    use tracing_error::ErrorLayer;
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::{fmt, EnvFilter};

    let fmt_layer = fmt::layer().with_thread_ids(true).with_target(true);
    let filter_layer = EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("warn"))?;

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .with(ErrorLayer::default())
        .init();

    Ok(())
}
