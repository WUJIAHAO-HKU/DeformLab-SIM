// Explicit initialization-only SDK controls for causal experiments; no state repair.
#include <pxr/usd/sdf/path.h>
#include <omni/physx/IPhysx.h>
#include <PxDeformableVolume.h>
#include <PxShape.h>
#include <PxDeformableVolumeMaterial.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
namespace py=pybind11;
PYBIND11_MODULE(physx_deformable_probe_controls,module) {
    module.def("configure",[](omni::physx::IPhysx* interface,const std::string& path,int model,bool disable_partial){
        if (!interface || model < -1 || model > 1) throw std::runtime_error("Invalid interface or material model");
        auto* volume=static_cast<physx::PxDeformableVolume*>(interface->getPhysXPtr(PXR_NS::SdfPath(path),omni::physx::ePTDeformableVolume));
        if (!volume) throw std::runtime_error("No live volume");
        if (disable_partial) volume->setDeformableVolumeFlag(physx::PxDeformableVolumeFlag::ePARTIALLY_KINEMATIC,false);
        if (model>=0) {
            physx::PxDeformableVolumeMaterial* material=nullptr;
            if (!volume->getShape() || volume->getShape()->getDeformableVolumeMaterials(&material,1)!=1 || !material) throw std::runtime_error("Missing volume material");
            material->setMaterialModel(static_cast<physx::PxDeformableVolumeMaterialModel::Enum>(model));
        }
    },py::arg("interface"),py::arg("path"),py::arg("model")=-1,py::arg("disable_partial")=false);
}
