#pragma once
#include "Interfaces/IVisualizationService.h"
#include "Types/LiverAITypes.h"

class LIVERIMAGEAI_API FVisualizationService : public IVisualizationService
{
public:
    FVisualizationService();
    virtual ~FVisualizationService();
    
    // IVisualizationService interface
    virtual void GenerateMeshFromData(const TArray<FMeshData>& MeshData, const FString& ActorName) override;
    virtual void GenerateTestMesh() override;
    virtual void ClearVisualization() override;
    virtual void FocusOnGeneratedMesh() override;
    virtual bool HasActiveMesh() const override;
    
    FOnMeshGenerated OnMeshGenerated;
    FOnVisualizationCleared OnVisualizationCleared;
    
private:
    TArray<class AActor*> SpawnedActors;
    
    class UProceduralMeshComponent* CreateMeshComponent(
        class AActor* Actor,
        const FMeshData& MeshData);
    
    void ApplyMaterialToMesh(
        class UProceduralMeshComponent* MeshComp,
        const FLinearColor& Color);
    
    class AActor* CreateMeshActor(const FString& ActorName);
    void DestroyAllActors();
};