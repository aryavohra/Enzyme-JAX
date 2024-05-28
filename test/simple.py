from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import jax.lax
import enzyme_ad.jax as enzyme_jax
import numpy as np
import timeit

argv = ("-I/usr/include/c++/11", "-I/usr/include/x86_64-linux-gnu/c++/11")

pipeline = enzyme_jax.JaXPipeline()
# pipeline = enzyme_jax.NewXLAPipeline(mlirad=False)

def test(x, y):
    return x + y


class Simple(absltest.TestCase):
    def test_simple_random(self):
        jfunc = jax.jit(test)

        efunc = jax.jit(
            enzyme_jax.enzyme_jax_ir(pipeline_options=enzyme_jax.JaXPipeline("equality-saturation-pass"),)(test)
        )

        eres = efunc(3, 5)
        print("enzyme forward", eres)



if __name__ == "__main__":
    absltest.main()
