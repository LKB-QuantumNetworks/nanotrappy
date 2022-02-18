import timeit
from atomicsystem import *
from arc import Cesium


at = atomicsystem(Cesium(), "6P3/2", f=4)

print(at.alpha_scalar(685e-9))
print(at.alpha_vector(685e-9))
print(at.alpha_tensor(685e-9))

# for key in at.dicoatom:
#     print(f"State{key} : {at.dicoatom[key]}")

# print(timeit.timeit(lambda: at.alpha_scalar(685e-9), number=100))


# print("List levels")
# print(len(at.listlevels))
# print("Couple levels")
# print(len(at.coupledlevels))

# for l, c in zip(at.listlevels, at.coupledlevels):
#     print(f"Level:{l}, Coupled: {c}")

# for l in at.listlevels:
#     print(f"Level:{l}")


# check = True
# for key in at.dicoatom:
#     if key in at.dicoatoms:
#         if at.dicoatoms[key] == at.dicoatom[key]:
#             continue
#         else:
#             check = False
#             print(key)
#             print(at.dicoatoms[key])
#             print(at.dicoatom[key])
#             break
# print(check)
