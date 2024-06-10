import jax.numpy as jnp
import jax.random
import jax.lax
import enzyme_ad.jax as enzyme_jax

def square(x, y):
    return jnp.transpose((x * y + x - y) / (jnp.exp(x) * jnp.tanh(y)))
    # return jnp.exp(x)

sqjit = jax.jit(
    enzyme_jax.enzyme_jax_ir(
        pipeline_options=enzyme_jax.JaXPipeline(
            "equality-saturation-pass"
        )
    )(square)
)

a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = jnp.array([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])

print(sqjit(a, b))
