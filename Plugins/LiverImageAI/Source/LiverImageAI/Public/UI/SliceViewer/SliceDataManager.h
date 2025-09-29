#pragma once
#include "CoreMinimal.h"
#include "Types/LiverAITypes.h"

/**
 * Manages slice viewer data storage and access
 * Single Responsibility: Store and provide access to image and mask data
 */
class LIVERIMAGEAI_API FSliceDataManager
{
public:
    FSliceDataManager();
    ~FSliceDataManager();
    
    // Data management
    void SetImageData(const TArray<float>& InImageData, int32 Width, int32 Height, int32 Depth);
    void SetMaskData(const TArray<uint8>& InMaskData, int32 Width, int32 Height, int32 Depth);
    void ClearData();
    
    // Data access
    const TArray<float>& GetImageData() const { return ImageData; }
    const TArray<uint8>& GetMaskData() const { return MaskData; }
    
    // Dimensions
    int32 GetWidth() const { return ImageWidth; }
    int32 GetHeight() const { return ImageHeight; }
    int32 GetDepth() const { return ImageDepth; }
    bool HasValidData() const { return ImageData.Num() > 0; }
    bool HasMaskData() const { return MaskData.Num() > 0; }
    
    // Voxel access
    float GetVoxelValue(int32 X, int32 Y, int32 Z) const;
    uint8 GetMaskValue(int32 X, int32 Y, int32 Z) const;
    int32 GetVoxelIndex(int32 X, int32 Y, int32 Z) const;
    
    // Statistics
    void GetDataRange(float& OutMin, float& OutMax) const;
    TMap<uint8, int32> GetMaskStatistics() const;
    
private:
    TArray<float> ImageData;
    TArray<uint8> MaskData;
    int32 ImageWidth;
    int32 ImageHeight;
    int32 ImageDepth;
    
    bool ValidateDataSize(int32 DataSize, int32 Width, int32 Height, int32 Depth) const;
};
