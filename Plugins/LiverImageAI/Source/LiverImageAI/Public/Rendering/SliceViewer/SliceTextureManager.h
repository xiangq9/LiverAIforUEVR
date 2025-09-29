#pragma once
#include "CoreMinimal.h"
#include "Engine/Texture2DDynamic.h"
#include "Styling/SlateBrush.h"

/**
 * Manages texture creation and updates for slice visualization
 * Single Responsibility: Create and manage textures for UI display
 */
class LIVERIMAGEAI_API FSliceTextureManager
{
public:
    FSliceTextureManager();
    ~FSliceTextureManager();
    
    // Texture management
    void CreateOrUpdateTexture(int32 ViewPlane, int32 Width, int32 Height, int32 Depth);
    void UpdateTextureData(const TArray<uint8>& PixelData);
    void ClearTexture();
    
    // Texture access
    UTexture2DDynamic* GetTexture() const { return SliceTexture; }
    const FSlateBrush* GetSlateBrush() const { return SliceBrush.Get(); }
    bool HasValidTexture() const { return SliceTexture != nullptr; }
    
    // Texture properties
    void GetTextureSize(int32& OutWidth, int32& OutHeight) const;
    int32 GetTextureWidth() const { return SliceTexture ? SliceTexture->SizeX : 0; }
    int32 GetTextureHeight() const { return SliceTexture ? SliceTexture->SizeY : 0; }
    
    // Display settings
    void SetTextureFilter(TextureFilter Filter);
    void SetDrawAsImage() { if (SliceBrush.IsValid()) SliceBrush->DrawAs = ESlateBrushDrawType::Image; }
    
private:
    UTexture2DDynamic* SliceTexture;
    TSharedPtr<FSlateBrush> SliceBrush;
    
    // Helper methods
    void CreateTexture(int32 Width, int32 Height);
    void UpdateTextureOnRenderThread(const TArray<uint8>& PixelData);
    void CalculateTextureDimensions(int32 ViewPlane, int32 Width, int32 Height, int32 Depth,
                                   int32& OutTextureWidth, int32& OutTextureHeight) const;
};
