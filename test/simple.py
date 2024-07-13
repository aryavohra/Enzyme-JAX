from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import jax.lax
import enzyme_ad.jax as enzyme_jax

def test(x, y):
    return jnp.matmul(jnp.transpose(jnp.transpose(x)), jnp.transpose(y))

class Simple(absltest.TestCase):
    def test_simple_random(self):
        jfunc = jax.jit(test)

        efunc = enzyme_jax.enzyme_jax_ir(pipeline_options=enzyme_jax.JaXPipeline("equality-saturation-pass"),)(test)
        
        a = jnp.array([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.0]])
        b = jnp.array([[6.0, 5.0, 4.0, 0.0], [3.0, 2.0, 1.0, 0.0]])
        
        eres = efunc(a, b)
        print("enzyme forward", eres)

if __name__ == "__main__":
    absltest.main()
