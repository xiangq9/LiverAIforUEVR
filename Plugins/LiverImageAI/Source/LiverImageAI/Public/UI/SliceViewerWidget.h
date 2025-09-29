#pragma once
#include "Widgets/SCompoundWidget.h"

// Forward declarations
class SLiverSliceViewer;
class STextBlock;

// Delegate for slice synchronization
DECLARE_DELEGATE_OneParam(FOnSliceChanged, int32);

class LIVERIMAGEAI_API SSliceViewerWidget : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SSliceViewerWidget) {}
        SLATE_ARGUMENT(FString, Title)
    SLATE_END_ARGS()
    
    void Construct(const FArguments& InArgs);
    
    // Image data management
    void SetImageData(const TArray<float>& ImageData, int32 Width, int32 Height, int32 Depth);
    void SetMaskData(const TArray<uint8>& MaskData, int32 Width, int32 Height, int32 Depth);
    void ClearData();
    
    // View controls
    void SetViewPlane(int32 Plane);
    void SetCurrentSlice(int32 Slice);
    int32 GetCurrentSlice() const;
    int32 GetViewPlane() const;
    
    // Navigation
    void NextSlice();
    void PreviousSlice();
    int32 GetMaxSlices() const;
    
    // Display settings
    void SetWindowLevel(float WindowCenter, float WindowWidth);
    void SetMaskOpacity(float Opacity);
    void SetZoomFactor(float ZoomFactor);
    
    // Synchronization
    FOnSliceChanged OnSliceChanged;
    
private:
    // UI components
    TSharedPtr<SLiverSliceViewer> InternalSliceViewer;  // 使用原有的 SLiverSliceViewer
    TSharedPtr<STextBlock> TitleText;
    
    // Display settings
    FString Title;
};