#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <algorithm>
#include "mapping/Mapping.hpp"
#include "mapping/NearestNeighborMapping.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/SharedPointer.hpp"
#include "mesh/Vertex.hpp"
#include "testing/Testing.hpp"

using namespace precice;
using namespace precice::mesh;

static void NearestNeighborMapping_ConsistentNonIncremental(benchmark::State &state)
{
  for (auto _ : state) {
    int dimensions = 2;

    // Create mesh to map from
    PtrMesh inMesh(new Mesh("InMesh", dimensions, testing::nextMeshID()));
    PtrData inDataScalar   = inMesh->createData("InDataScalar", 1, 0);
    PtrData inDataVector   = inMesh->createData("InDataVector", 2, 1);
    int     inDataScalarID = inDataScalar->getID();
    int     inDataVectorID = inDataVector->getID();
    Vertex &inVertex0      = inMesh->createVertex(Eigen::Vector2d::Constant(0.0));
    Vertex &inVertex1      = inMesh->createVertex(Eigen::Vector2d::Constant(1.0));
    inMesh->allocateDataValues();
    Eigen::VectorXd &inValuesScalar = inDataScalar->values();
    Eigen::VectorXd &inValuesVector = inDataVector->values();
    inValuesScalar << 1.0, 2.0;
    inValuesVector << 1.0, 2.0, 3.0, 4.0;

    // Create mesh to map to
    PtrMesh outMesh(new Mesh("OutMesh", dimensions, testing::nextMeshID()));
    PtrData outDataScalar   = outMesh->createData("OutDataScalar", 1, 2);
    PtrData outDataVector   = outMesh->createData("OutDataVector", 2, 3);
    int     outDataScalarID = outDataScalar->getID();
    int     outDataVectorID = outDataVector->getID();
    Vertex &outVertex0      = outMesh->createVertex(Eigen::Vector2d::Constant(0.0));
    Vertex &outVertex1      = outMesh->createVertex(Eigen::Vector2d::Constant(1.0));
    outMesh->allocateDataValues();

    // Setup mapping with mapping coordinates and geometry used
    precice::mapping::NearestNeighborMapping mapping(mapping::Mapping::CONSISTENT, dimensions);
    mapping.setMeshes(inMesh, outMesh);

    // Map data with coinciding vertices, has to result in equal values.
    mapping.computeMapping();
    mapping.map(inDataScalarID, outDataScalarID);
    const Eigen::VectorXd &outValuesScalar = outDataScalar->values();
    mapping.map(inDataVectorID, outDataVectorID);
    const Eigen::VectorXd &outValuesVector = outDataVector->values();

    // Map data with almost coinciding vertices, has to result in equal values.
    inVertex0.setCoords(outVertex0.getCoords() + Eigen::Vector2d::Constant(0.1));
    inVertex1.setCoords(outVertex1.getCoords() + Eigen::Vector2d::Constant(0.1));
    mapping.computeMapping();
    mapping.map(inDataScalarID, outDataScalarID);
    mapping.map(inDataVectorID, outDataVectorID);

    // Map data with exchanged vertices, has to result in exchanged values.
    inVertex0.setCoords(outVertex1.getCoords());
    inVertex1.setCoords(outVertex0.getCoords());
    mapping.computeMapping();
    mapping.map(inDataScalarID, outDataScalarID);
    mapping.map(inDataVectorID, outDataVectorID);
    Eigen::Vector4d expected(3.0, 4.0, 1.0, 2.0);

    // Map data with coinciding output vertices, has to result in same values.
    outVertex1.setCoords(outVertex0.getCoords());
    mapping.computeMapping();
    mapping.map(inDataScalarID, outDataScalarID);
    mapping.map(inDataVectorID, outDataVectorID);
    expected << 3.0, 4.0, 3.0, 4.0;
  }
}

BENCHMARK(NearestNeighborMapping_ConsistentNonIncremental);