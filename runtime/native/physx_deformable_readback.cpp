// Read-only diagnostic bridge; compiled against the exact installed SDK headers.
#include <pxr/usd/sdf/path.h>
#include <omni/physx/IPhysx.h>
#include <PxDeformableVolume.h>
#include <PxShape.h>
#include <PxScene.h>
#include <PxRigidActor.h>
#include <vector>
#include <PxDeformableVolumeMaterial.h>
#include <geometry/PxTetrahedronMesh.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

PYBIND11_MODULE(physx_deformable_readback, module)
{
    module.def("simulation_buffer", [](omni::physx::IPhysx* interface, const std::string& path) {
        if (!interface) throw std::runtime_error("Null PhysX interface");
        auto* volume = static_cast<physx::PxDeformableVolume*>(
            interface->getPhysXPtr(PXR_NS::SdfPath(path), omni::physx::ePTDeformableVolume));
        if (!volume) throw std::runtime_error("No live deformable volume at path");
        auto* mesh = volume->getSimulationMesh();
        auto* buffer = volume->getSimPositionInvMassBufferD();
        if (!mesh || !buffer) throw std::runtime_error("Missing simulation mesh or GPU buffer");
        py::dict result;
        result["device_pointer"] = reinterpret_cast<std::uintptr_t>(buffer);
        auto* velocity = volume->getSimVelocityBufferD();
        if (!velocity) throw std::runtime_error("Missing simulation velocity buffer");
        result["velocity_pointer"] = reinterpret_cast<std::uintptr_t>(velocity);
        result["node_count"] = mesh->getNbVertices();
        result["sdk_version"] = "5.9.1";
        py::dict settings;
        settings["linear_damping"] = volume->getLinearDamping();
        settings["settling_damping"] = volume->getSettlingDamping();
        settings["settling_threshold"] = volume->getSettlingThreshold();
        settings["sleep_threshold"] = volume->getSleepThreshold();
        settings["wake_counter"] = volume->getWakeCounter();
        settings["is_sleeping"] = volume->isSleeping();
        settings["max_linear_velocity"] = volume->getMaxLinearVelocity();
        settings["body_flags"] = static_cast<unsigned int>(volume->getDeformableBodyFlags());
        settings["volume_flags"] = static_cast<unsigned int>(volume->getDeformableVolumeFlags());
        physx::PxU32 positionIters=0, velocityIters=0;
        volume->getSolverIterationCounts(positionIters,velocityIters);
        settings["position_iterations"] = positionIters;
        settings["velocity_iterations"] = velocityIters;
        physx::PxDeformableVolumeMaterial* material=nullptr;
        if (volume->getShape() && volume->getShape()->getDeformableVolumeMaterials(&material,1)==1 && material) {
            settings["elasticity_damping"] = material->getElasticityDamping();
            settings["material_model"] = static_cast<int>(material->getMaterialModel());
            settings["youngs_modulus"] = material->getYoungsModulus();
            settings["poissons_ratio"] = material->getPoissons();
        }
        result["settings"] = settings;
        py::list actors_out;
        auto* scene=volume->getScene();
        if (scene) {
            auto types=physx::PxActorTypeFlag::eRIGID_STATIC | physx::PxActorTypeFlag::eRIGID_DYNAMIC;
            std::vector<physx::PxActor*> actors(scene->getNbActors(types));
            if (!actors.empty()) scene->getActors(types,actors.data(),static_cast<physx::PxU32>(actors.size()));
            for (auto* actor:actors) {
                auto* rigid=static_cast<physx::PxRigidActor*>(actor);
                std::vector<physx::PxShape*> shapes(rigid->getNbShapes());
                if (!shapes.empty()) rigid->getShapes(shapes.data(),static_cast<physx::PxU32>(shapes.size()));
                py::dict row;py::list shape_rows;
                row["name"]=actor->getName() ? actor->getName() : "";
                for (auto* shape:shapes) {
                    py::dict item;
                    item["geometry_type"]=static_cast<int>(shape->getGeometry().getType());
                    item["simulation_shape"]=bool(shape->getFlags() & physx::PxShapeFlag::eSIMULATION_SHAPE);
                    shape_rows.append(item);
                }
                row["shapes"]=shape_rows;actors_out.append(row);
            }
        }
        result["scene_rigid_shapes"]=actors_out;
        return result;
    }, py::arg("interface"), py::arg("path"));
}
