use std::path::Path;
use std::time::Duration;

use cgmath::{Deg, InnerSpace, Quaternion, Rotation3, Vector3, Zero};
use color_eyre::eyre::Result;
use color_eyre::Report;
use rand::Rng;
use tracing::instrument;

use recs_gfx::engine::{Creator, Engine, GenericResult, RingSender};
use recs_gfx::time::UpdateRate;
use recs_gfx::{egui, Object, Transform};

struct SimulationContext {
    objects: Vec<Object>,
}

impl SimulationContext {
    fn new() -> Self {
        SimulationContext { objects: vec![] }
    }

    fn animate_rotation(
        &mut self,
        sender: &mut RingSender<Vec<Object>>,
        delta_time: &Duration,
    ) -> Result<(), Report> {
        let rotation_delta = 10.0 * delta_time.as_secs_f32();

        let new_objects: Vec<Object> = self
            .objects
            .iter_mut()
            .map(|object| {
                let old_transform = object.transform;
                let rotation_axis = if old_transform.position.is_zero() {
                    Vector3::unit_z()
                } else {
                    old_transform.position.normalize()
                };
                let rotation = Quaternion::from_axis_angle(rotation_axis, Deg(rotation_delta));
                let new_transform = Transform {
                    rotation: old_transform.rotation * rotation,
                    ..old_transform
                };
                object.transform = new_transform;
                *object
            })
            .collect();

        sender.send(new_objects)?;

        Ok(())
    }
}

#[instrument]
fn main() -> Result<(), Report> {
    install_tracing()?;

    color_eyre::install()?;

    fn init_gfx(context: &mut SimulationContext, creator: &mut dyn Creator) -> GenericResult<()> {
        let model = creator.load_model(Path::new("cube.obj"))?;

        let transforms = create_transforms();
        context.objects = creator.create_objects(model, transforms)?;

        Ok(())
    }

    fn simulate(
        context: &mut SimulationContext,
        time: &UpdateRate,
        sender: &mut RingSender<Vec<Object>>,
    ) -> GenericResult<()> {
        context.animate_rotation(sender, &time.delta_time)?;
        Ok(())
    }

    let engine = Engine::new(init_gfx, Some(ui), simulate, SimulationContext::new())?;
    engine.start()?;

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

/// Example UI.
fn ui(ctx: &egui::Context) {
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Organize windows").clicked() {
                    ui.ctx().memory().reset_areas();
                    ui.close_menu();
                }
                if ui
                    .button("Reset egui memory")
                    .on_hover_text("Forget scroll, positions, sizes etc")
                    .clicked()
                {
                    *ui.ctx().memory() = Default::default();
                    ui.close_menu();
                }
            });
        });
    });
    egui::Window::new("Test").resizable(true).show(ctx, |ui| {
        let _test_button = ui.button("test button");
        ui.allocate_space(ui.available_size())
    });
}

fn install_tracing() -> Result<(), Report> {
    use tracing_error::ErrorLayer;
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::{fmt, EnvFilter};

    let fmt_layer = fmt::layer().with_thread_ids(true).with_target(false);
    let filter_layer = EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("warn"))?;

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .with(ErrorLayer::default())
        .init();

    Ok(())
}
