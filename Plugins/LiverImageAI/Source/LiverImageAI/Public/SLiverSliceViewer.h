#pragma once

#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"
#include "Widgets/Images/SImage.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Input/SButton.h"
#include "Styling/SlateBrush.h"
#include "Engine/Texture2DDynamic.h"

// Delegate for slice change notification
DECLARE_DELEGATE_OneParam(FOnSliceChanged, int32);

/**
 * Medical image slice viewer for CT/MRI visualization with mask overlay support
 */
class LIVERIMAGEAI_API SLiverSliceViewer : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SLiverSliceViewer) {}
    SLATE_END_ARGS()

    void Construct(const FArguments& InArgs);

    // Core functionality
    void SetImageData(const TArray<float>& InImageData, int32 Width, int32 Height, int32 Depth);
    void SetMaskData(const TArray<uint8>& InMaskData, int32 Width, int32 Height, int32 Depth);
    void ClearData();

    // Navigation
    void NextSlice();
    void PreviousSlice();
    void SetCurrentSlice(int32 SliceIndex);
    int32 GetCurrentSlice() const { return CurrentSlice; }
    int32 GetMaxSlices() const;

    // View control
    void SetViewPlane(int32 Plane);
    int32 GetViewPlane() const { return ViewPlane; }

    // Mouse interaction
    virtual FReply OnMouseWheel(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent) override;

    // Advanced features
    void SetWindowLevel(float WindowCenter, float WindowWidth);
    void SetMaskOpacity(float Opacity);
    void SetZoomFactor(float ZoomFactor);

    // Synchronization
    FOnSliceChanged OnSliceChanged;

private:
    // UI components
    TSharedPtr<SImage> SliceImage;
    TSharedPtr<FSlateBrush> SliceBrush;
    TSharedPtr<STextBlock> SliceInfoText;
    TSharedPtr<SButton> PreviousButton;
    TSharedPtr<SButton> NextButton;

    // Data storage
    TArray<float> ImageData;
    TArray<uint8> MaskData;
    UTexture2DDynamic* SliceTexture;
    
    // Image properties
    int32 ImageWidth;
    int32 ImageHeight;
    int32 ImageDepth;
    int32 CurrentSlice;
    int32 ViewPlane;
    
    // Display settings
    float WindowCenter;
    float WindowWidth;
    float ZoomFactor;
    float MaskOpacity;
    bool bShowMask;
    
    // Interaction state
    bool bIsPanning;
    FVector2D LastMousePosition;
    FVector2D PanOffset;

    // Internal methods
    void CreateOrUpdateTexture();
    void UpdateSliceTexture();
    void ExtractSliceData(TArray<uint8>& OutPixelData, int32 TextureWidth, int32 TextureHeight);
    uint8 ApplyWindowLevel(float Intensity) const;
    FLinearColor GetMaskColor(uint8 ClassId) const;
    int32 GetVoxelIndex(int32 X, int32 Y, int32 Slice, int32 PlaneType) const;
    int32 ClampSliceIndex(int32 SliceIndex) const;

    // UI callbacks
    FReply OnPreviousClicked();
    FReply OnNextClicked();
    FText GetSliceText() const;
    FText GetViewPlaneText() const;

    // Utility
    void UpdateSliceInfo();
    void RefreshDisplay();
};