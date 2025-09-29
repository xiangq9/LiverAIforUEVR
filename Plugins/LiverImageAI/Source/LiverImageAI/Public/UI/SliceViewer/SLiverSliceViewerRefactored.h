#pragma once
#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"
#include "UI/SliceViewer/SliceNavigator.h"

// Forward declarations
class FSliceDataManager;
class FSliceRenderer;
class FSliceTextureManager;
class SImage;
class STextBlock;
class SButton;

/**
 * Refactored medical image slice viewer using modular components
 * Follows Single Responsibility Principle - acts as UI coordinator
 */
class LIVERIMAGEAI_API SLiverSliceViewerRefactored : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SLiverSliceViewerRefactored) {}
    SLATE_END_ARGS()

    void Construct(const FArguments& InArgs);
    virtual ~SLiverSliceViewerRefactored();

    // Public interface - delegates to appropriate components
    void SetImageData(const TArray<float>& InImageData, int32 Width, int32 Height, int32 Depth);
    void SetMaskData(const TArray<uint8>& InMaskData, int32 Width, int32 Height, int32 Depth);
    void ClearData();
    
    // Navigation interface
    void NextSlice();
    void PreviousSlice();
    void SetCurrentSlice(int32 SliceIndex);
    int32 GetCurrentSlice() const;
    int32 GetMaxSlices() const;
    
    // View control interface
    void SetViewPlane(int32 Plane);
    int32 GetViewPlane() const;
    
    // Display settings interface
    void SetWindowLevel(float WindowCenter, float WindowWidth);
    void SetMaskOpacity(float Opacity);
    void SetZoomFactor(float ZoomFactor);
    void ShowMask(bool bShow);
    
    // Events
    FOnSliceChanged OnSliceChanged;
    
    // Mouse interaction override
    virtual FReply OnMouseWheel(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent) override;
    virtual FReply OnMouseButtonDown(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent) override;
    virtual FReply OnMouseButtonUp(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent) override;
    virtual FReply OnMouseMove(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent) override;
    
private:
    // Core components - composition over inheritance
    TSharedPtr<FSliceDataManager> DataManager;
    TSharedPtr<FSliceNavigator> Navigator;
    TSharedPtr<FSliceRenderer> Renderer;
    TSharedPtr<FSliceTextureManager> TextureManager;
    
    // UI components
    TSharedPtr<SImage> SliceImage;
    TSharedPtr<STextBlock> SliceInfoText;
    TSharedPtr<SButton> PreviousButton;
    TSharedPtr<SButton> NextButton;
    
    // Interaction state
    bool bIsPanning;
    FVector2D LastMousePosition;
    FVector2D PanOffset;
    float ZoomFactor;
    
    // Internal methods
    void InitializeComponents();
    void BuildUI();
    void UpdateDisplay();
    void UpdateSliceInfo();
    void ConnectEventHandlers();
    
    // UI callbacks
    FReply OnPreviousClicked();
    FReply OnNextClicked();
    FText GetSliceText() const;
    
    // Component event handlers
    void OnInternalSliceChanged(int32 NewSlice);
    void OnViewPlaneChanged(int32 NewPlane);
};
