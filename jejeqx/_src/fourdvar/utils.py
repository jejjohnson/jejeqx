import kernex as kex


@kex.kmap(kernel_size=(2,), relative=True)
def time_patches(x):
    return x
