pub use nalgebra as na;
use na::Vector2;

pub fn search(
    init: Vector2<f64>, eps: f64, 
    f: impl Fn(Vector2<f64>) -> f64, 
    grad: impl Fn(Vector2<f64>) -> Vector2<f64>,
    search_lam: impl Fn(&dyn Fn(f64) -> f64) -> f64,
) -> Vector2<f64> {
    
    assert!(eps > 0.0);

    let mut x = init;
    let mut gradx = grad(x);
    while grad(x).norm() > eps {
        let nextx = |lam: f64| x - gradx * lam;
        let lam = search_lam(&|lam: f64| f(nextx(lam)));
        x = nextx(lam);
        gradx = grad(x);
    }
    
    x
}
