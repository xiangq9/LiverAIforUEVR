#include "Rendering/VisualizationService.h"
#include "ProceduralMeshComponent.h"
#include "Engine/World.h"
#include "Editor.h"
#include "Components/SceneComponent.h"
#include "Materials/Material.h"
#include "Materials/MaterialInstanceDynamic.h"

FVisualizationService::FVisualizationService()
{
}

FVisualizationService::~FVisualizationService()
{
    ClearVisualization();
}

void FVisualizationService::GenerateMeshFromData(const TArray<FMeshData>& MeshData, const FString& ActorName)
{
    if (MeshData.Num() == 0)
    {
        return;
    }
    
    AActor* NewActor = CreateMeshActor(ActorName);
    if (!NewActor)
    {
        return;
    }
    
    for (const FMeshData& Mesh : MeshData)
    {
        UProceduralMeshComponent* MeshComp = CreateMeshComponent(NewActor, Mesh);
        if (MeshComp)
        {
            ApplyMaterialToMesh(MeshComp, Mesh.Color);
        }
    }
    
    SpawnedActors.Add(NewActor);
    FocusOnGeneratedMesh();
    OnMeshGenerated.Broadcast(ActorName);
}

void FVisualizationService::GenerateTestMesh()
{
    // Create test cube mesh data
    FMeshData TestMesh;
    TestMesh.OrganName = TEXT("TestCube");
    TestMesh.Color = FLinearColor(0.8f, 0.2f, 0.2f, 1.0f);
    
    // Define cube vertices
    TestMesh.Vertices = {
        FVector(-50, -50, -50), FVector(50, -50, -50),
        FVector(50, 50, -50), FVector(-50, 50, -50),
        FVector(-50, -50, 50), FVector(50, -50, 50),
        FVector(50, 50, 50), FVector(-50, 50, 50)
    };
    
    // Define triangles
    TestMesh.Triangles = {
        0,1,2, 2,3,0,  // Bottom
        4,7,6, 6,5,4,  // Top
        0,4,5, 5,1,0,  // Front
        2,6,7, 7,3,2,  // Back
        0,3,7, 7,4,0,  // Left
        1,5,6, 6,2,1   // Right
    };
    
    TArray<FMeshData> MeshArray;
    MeshArray.Add(TestMesh);
    GenerateMeshFromData(MeshArray, TEXT("TestCube"));
}

void FVisualizationService::ClearVisualization()
{
    DestroyAllActors();
    OnVisualizationCleared.Broadcast();
}

void FVisualizationService::FocusOnGeneratedMesh()
{
    if (SpawnedActors.Num() > 0 && GEditor)
    {
        AActor* LastActor = SpawnedActors.Last();
        GEditor->SelectNone(false, true);
        GEditor->SelectActor(LastActor, true, true);
        GEditor->NoteSelectionChange();
        GEditor->MoveViewportCamerasToActor(*LastActor, false);
    }
}

bool FVisualizationService::HasActiveMesh() const
{
    return SpawnedActors.Num() > 0;
}

AActor* FVisualizationService::CreateMeshActor(const FString& ActorName)
{
    UWorld* World = nullptr;
    if (GEditor)
    {
        World = GEditor->GetEditorWorldContext().World();
    }
    
    if (!World)
    {
        return nullptr;
    }
    
    FActorSpawnParameters SpawnParams;
    SpawnParams.Name = *ActorName;
    
    AActor* NewActor = World->SpawnActor<AActor>(
        AActor::StaticClass(),
        FVector::ZeroVector,
        FRotator::ZeroRotator,
        SpawnParams);
    
    if (NewActor)
    {
        USceneComponent* RootComp = NewObject<USceneComponent>(NewActor, TEXT("RootComponent"));
        NewActor->SetRootComponent(RootComp);
        RootComp->RegisterComponent();
    }
    
    return NewActor;
}

UProceduralMeshComponent* FVisualizationService::CreateMeshComponent(AActor* Actor, const FMeshData& MeshData)
{
    if (!Actor || MeshData.Vertices.Num() == 0 || MeshData.Triangles.Num() == 0)
    {
        return nullptr;
    }
    
    FString CompName = FString::Printf(TEXT("%s_Mesh"), *MeshData.OrganName);
    UProceduralMeshComponent* MeshComp = NewObject<UProceduralMeshComponent>(Actor, *CompName);
    
    if (MeshComp)
    {
        MeshComp->AttachToComponent(Actor->GetRootComponent(), FAttachmentTransformRules::KeepRelativeTransform);
        MeshComp->RegisterComponent();
        
        // Generate normals and other vertex data
        TArray<FVector> Normals;
        TArray<FVector2D> UV0;
        TArray<FLinearColor> VertexColors;
        TArray<FProcMeshTangent> Tangents;
        
        Normals.Init(FVector::UpVector, MeshData.Vertices.Num());
        UV0.Init(FVector2D::ZeroVector, MeshData.Vertices.Num());
        VertexColors.Init(MeshData.Color, MeshData.Vertices.Num());
        
        MeshComp->CreateMeshSection_LinearColor(
            0,
            MeshData.Vertices,
            MeshData.Triangles,
            Normals,
            UV0,
            VertexColors,
            Tangents,
            true
        );
        
        MeshComp->SetCollisionEnabled(ECollisionEnabled::QueryOnly);
    }
    
    return MeshComp;
}

void FVisualizationService::ApplyMaterialToMesh(UProceduralMeshComponent* MeshComp, const FLinearColor& Color)
{
    if (!MeshComp)
    {
        return;
    }
    
    UMaterial* DefaultMaterial = Cast<UMaterial>(
        StaticLoadObject(UMaterial::StaticClass(), nullptr, 
        TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial")));
    
    if (DefaultMaterial)
    {
        UMaterialInstanceDynamic* DynMaterial = UMaterialInstanceDynamic::Create(DefaultMaterial, MeshComp);
        if (DynMaterial)
        {
            DynMaterial->SetVectorParameterValue(TEXT("Color"), Color);
            MeshComp->SetMaterial(0, DynMaterial);
        }
    }
}

void FVisualizationService::DestroyAllActors()
{
    for (AActor* Actor : SpawnedActors)
    {
        if (IsValid(Actor))
        {
            Actor->Destroy();
        }
    }
    SpawnedActors.Empty();
}