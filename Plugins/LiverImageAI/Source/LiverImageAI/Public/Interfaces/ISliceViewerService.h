#pragma once
#include "CoreMinimal.h"

class LIVERIMAGEAI_API ISliceViewerService
{
public:
    virtual ~ISliceViewerService() = default;
    
    DECLARE_DELEGATE_OneParam(FOnSliceChanged, int32);
    
    virtual void SetImageData(const TArray<float>& ImageData, int32 Width, int32 Height, int32 Depth) = 0;
    virtual void SetMaskData(const TArray<uint8>& MaskData, int32 Width, int32 Height, int32 Depth) = 0;
    virtual void SetCurrentSlice(int32 SliceIndex) = 0;
    virtual int32 GetCurrentSlice() const = 0;
    virtual void SetViewPlane(int32 Plane) = 0;
    virtual int32 GetViewPlane() const = 0;
    virtual void ClearData() = 0;
};