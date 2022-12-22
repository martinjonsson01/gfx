use log::error;
use recs_gfx::run;
use std::error::Error;

fn main() {
    if let Err(error) = pollster::block_on(run()) {
        print_error(&error, false, 0);
    }
}

fn print_error(error: &dyn Error, is_source: bool, indent_level: usize) {
    let indents = "  ".repeat(indent_level);
    let due_to = if is_source { "caused by: " } else { "" };
    error!("{indents}{due_to}{error}");
    if let Some(source) = error.source() {
        print_error(source, true, indent_level + 1);
    }
}
