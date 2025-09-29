#pragma once
#include "CoreMinimal.h"
#include "Types/LiverAITypes.h"

class LIVERIMAGEAI_API IVisualizationService
{
public:
    virtual ~IVisualizationService() = default;
    
    DECLARE_MULTICAST_DELEGATE_OneParam(FOnMeshGenerated, const FString&);
    DECLARE_MULTICAST_DELEGATE(FOnVisualizationCleared);
    
    virtual void GenerateMeshFromData(const TArray<FMeshData>& MeshData, const FString& ActorName) = 0;
    virtual void GenerateTestMesh() = 0;
    virtual void ClearVisualization() = 0;
    virtual void FocusOnGeneratedMesh() = 0;
    virtual bool HasActiveMesh() const = 0;
};