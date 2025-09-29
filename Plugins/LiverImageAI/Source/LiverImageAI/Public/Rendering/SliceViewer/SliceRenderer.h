#pragma once
#include "CoreMinimal.h"
#include "Engine/Texture2DDynamic.h"

class FSliceDataManager;
class FSliceNavigator;

/**
 * Handles slice rendering operations
 * Single Responsibility: Extract and prepare slice data for display
 */
class LIVERIMAGEAI_API FSliceRenderer
{
public:
    FSliceRenderer();
    ~FSliceRenderer();
    
    // Rendering operations
    void ExtractSliceData(TArray<uint8>& OutPixelData,
                         const FSliceDataManager& DataManager,
                         const FSliceNavigator& Navigator);
    
    // Window/Level settings
    void SetWindowLevel(float WindowCenter, float WindowWidth);
    float GetWindowCenter() const { return WindowCenter; }
    float GetWindowWidth() const { return WindowWidth; }
    void AutoCalculateWindowLevel(const TArray<float>& ImageData);
    
    // Mask overlay settings
    void SetMaskOpacity(float Opacity) { MaskOpacity = FMath::Clamp(Opacity, 0.0f, 1.0f); }
    void SetShowMask(bool bShow) { bShowMask = bShow; }
    float GetMaskOpacity() const { return MaskOpacity; }
    bool IsShowingMask() const { return bShowMask; }
    
    // Color mapping
    FLinearColor GetMaskColor(uint8 ClassId) const;
    void SetMaskColor(uint8 ClassId, const FLinearColor& Color);
    
    // Display settings
    void SetGamma(float Gamma) { DisplayGamma = FMath::Clamp(Gamma, 0.1f, 3.0f); }
    void SetInvertColors(bool bInvert) { bInvertColors = bInvert; }
    
private:
    float WindowCenter;
    float WindowWidth;
    float MaskOpacity;
    float DisplayGamma;
    bool bShowMask;
    bool bInvertColors;
    
    TMap<uint8, FLinearColor> MaskColorMap;
    
    // Helper methods
    uint8 ApplyWindowLevel(float Value, float DataMin, float DataMax) const;
    uint8 ApplyGamma(uint8 Value) const;
    void BlendMaskColor(uint8& R, uint8& G, uint8& B, uint8 GrayValue, const FLinearColor& MaskColor) const;
    void InitializeDefaultColors();
};
