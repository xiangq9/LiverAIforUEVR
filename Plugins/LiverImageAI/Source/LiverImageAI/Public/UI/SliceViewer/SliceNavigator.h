#pragma once
#include "CoreMinimal.h"

DECLARE_DELEGATE_OneParam(FOnSliceChanged, int32);
DECLARE_DELEGATE_OneParam(FOnViewPlaneChanged, int32);

/**
 * Handles slice navigation and view plane management
 * Single Responsibility: Manage current slice position and navigation
 */
class LIVERIMAGEAI_API FSliceNavigator
{
public:
    FSliceNavigator();
    ~FSliceNavigator();
    
    // Navigation
    void NextSlice();
    void PreviousSlice();
    void SetCurrentSlice(int32 SliceIndex);
    void JumpToSlice(int32 SliceIndex);
    int32 GetCurrentSlice() const { return CurrentSlice; }
    
    // View plane management
    void SetViewPlane(int32 Plane);
    int32 GetViewPlane() const { return ViewPlane; }
    FText GetViewPlaneText() const;
    
    // Dimensions
    void SetDimensions(int32 Width, int32 Height, int32 Depth);
    void GetDimensions(int32& OutWidth, int32& OutHeight, int32& OutDepth) const;
    
    // Slice info
    int32 GetMaxSlices() const;
    FText GetSliceInfoText() const;
    bool CanNavigateNext() const;
    bool CanNavigatePrevious() const;
    
    // Mouse wheel navigation
    void HandleMouseWheel(float WheelDelta);
    void SetMouseWheelSensitivity(float Sensitivity) { MouseWheelSensitivity = Sensitivity; }
    
    // Events
    FOnSliceChanged OnSliceChanged;
    FOnViewPlaneChanged OnViewPlaneChanged;
    
private:
    int32 CurrentSlice;
    int32 ViewPlane; // 0=Axial, 1=Coronal, 2=Sagittal
    int32 ImageWidth;
    int32 ImageHeight;
    int32 ImageDepth;
    float MouseWheelSensitivity;
    
    int32 ClampSliceIndex(int32 SliceIndex) const;
    void NotifySliceChanged();
    void NotifyViewPlaneChanged();
};
