from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import jax.lax
import enzyme_ad.jax as enzyme_jax

def test(x, y):
    return x + y + x + y + x + y + x + y

class Simple(absltest.TestCase):
    def test_simple_random(self):
        jfunc = jax.jit(test)

        efunc = enzyme_jax.enzyme_jax_ir(pipeline_options=enzyme_jax.JaXPipeline("equality-saturation-pass"),)(test)
        
        ka, kb, kc, kd = jax.random.split(jax.random.PRNGKey(0), num=4)
        a = jax.random.uniform(ka, shape=(2,))
        b = jax.random.uniform(kb, shape=(2,))
        c = jax.random.uniform(kc, shape=(2, 2, 2, 2))
        d = jax.random.uniform(kd, shape=(2, 2, 2, 2))

        eres = efunc(a, b)
        print("enzyme forward", eres)

if __name__ == "__main__":
    absltest.main()
