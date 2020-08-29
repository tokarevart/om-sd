pub use nalgebra as na;

macro_rules! make_search_fn {
    ($name:ident, $argtype:ty) => {
        pub fn $name(
            init: $argtype, eps: f64, 
            f: impl Fn($argtype) -> f64, 
            grad: impl Fn($argtype) -> $argtype,
            search_lam: impl Fn(&dyn Fn(f64) -> f64) -> f64,
        ) -> $argtype {
            
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
    };
}

make_search_fn!(search_2d, na::Vector2<f64>);
make_search_fn!(search_3d, na::Vector3<f64>);
