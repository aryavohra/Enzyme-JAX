from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import jax.lax
import enzyme_ad.jax as enzyme_jax

def test(x, y, z, w):
    dims = len(jnp.shape(x))
    a = jnp.concat([x, z], axis=2)
    b = jnp.concat([y, w], axis=2)
    numbers = (([1, 2], [0, 2]), ([0], [3]))
    res1 = jax.lax.dot_general(a, b, numbers)
    res2 = jax.lax.dot_general(x, y, numbers) + jax.lax.dot_general(z, w, numbers)
    print("RES1", res1) 
    print("RES2", res2)
    return res1 == res2

class Simple(absltest.TestCase):
    def test_simple_random(self):
        jfunc = jax.jit(test)

        efunc = enzyme_jax.enzyme_jax_ir(pipeline_options=enzyme_jax.JaXPipeline("equality-saturation-pass"),)(test)
        
        ka, kb, kc, kd = jax.random.split(jax.random.PRNGKey(0), num=4)
        a = jax.random.uniform(ka, shape=(2, 2, 2, 2))
        b = jax.random.uniform(kb, shape=(2, 2, 2, 2))
        c = jax.random.uniform(kc, shape=(2, 2, 2, 2))
        d = jax.random.uniform(kd, shape=(2, 2, 2, 2))

        eres = efunc(a, b, c, d)
        print("enzyme forward", eres)

if __name__ == "__main__":
    absltest.main()
