#include "SLiverSliceViewer.h"
#include "LiverAITypes.h"  // Add this line for LogLiverAI declaration
#include "Engine/Texture2DDynamic.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Text/STextBlock.h"
#include "Framework/Application/SlateApplication.h"
#include "RenderUtils.h"
#include "Engine/Engine.h"
#include "RHI.h"
#include "RHICommandList.h"
#include "Styling/SlateBrush.h"
#include "Algo/MinElement.h"
#include "Algo/MaxElement.h"

void SLiverSliceViewer::Construct(const FArguments& InArgs)
{
    // Initialize all member variables
    CurrentSlice = 0;
    ViewPlane = 0;
    ImageWidth = 0;
    ImageHeight = 0;
    ImageDepth = 0;
    SliceTexture = nullptr;
    
    // Initialize display settings
    WindowCenter = 0.5f;
    WindowWidth = 1.0f;
    ZoomFactor = 1.0f;
    MaskOpacity = 0.5f;
    bShowMask = true;
    
    // Initialize interaction state
    bIsPanning = false;
    LastMousePosition = FVector2D::ZeroVector;
    PanOffset = FVector2D::ZeroVector;
    
    ChildSlot
    [
        SNew(SVerticalBox)
        + SVerticalBox::Slot()
        .FillHeight(1.0f)
        [
            SNew(SBorder)
            .BorderImage(FCoreStyle::Get().GetBrush("ToolPanel.DarkGroupBorder"))
            .Padding(4)
            [
                SNew(SBox)
                .MinDesiredWidth(400)
                .MinDesiredHeight(400)
                [
                    SAssignNew(SliceImage, SImage)
                    .ColorAndOpacity(FLinearColor::White)
                ]
            ]
        ]
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5)
        [
            SNew(SHorizontalBox)
            + SHorizontalBox::Slot()
            .AutoWidth()
            .Padding(2)
            [
                SAssignNew(PreviousButton, SButton)
                .Text(FText::FromString(TEXT("<")))
                .OnClicked(this, &SLiverSliceViewer::OnPreviousClicked)
                .ToolTipText(FText::FromString(TEXT("Previous Slice")))
            ]
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .HAlign(HAlign_Center)
            .VAlign(VAlign_Center)
            [
                SAssignNew(SliceInfoText, STextBlock)
                .Text(this, &SLiverSliceViewer::GetSliceText)
                .Justification(ETextJustify::Center)
            ]
            + SHorizontalBox::Slot()
            .AutoWidth()
            .Padding(2)
            [
                SAssignNew(NextButton, SButton)
                .Text(FText::FromString(TEXT(">")))
                .OnClicked(this, &SLiverSliceViewer::OnNextClicked)
                .ToolTipText(FText::FromString(TEXT("Next Slice")))
            ]
        ]
    ];
}

void SLiverSliceViewer::SetImageData(const TArray<float>& InImageData, int32 Width, int32 Height, int32 Depth)
{
    UE_LOG(LogLiverAI, Warning, TEXT("=== SetImageData called ==="));
    UE_LOG(LogLiverAI, Warning, TEXT("Input dimensions: %d x %d x %d"), Width, Height, Depth);
    UE_LOG(LogLiverAI, Warning, TEXT("Input data size: %d"), InImageData.Num());
    UE_LOG(LogLiverAI, Warning, TEXT("Expected data size: %d"), Width * Height * Depth);
    
    if (InImageData.Num() != Width * Height * Depth)
    {
        UE_LOG(LogLiverAI, Error, TEXT("❌ Data size mismatch! Expected: %d, Got: %d"), 
            Width * Height * Depth, InImageData.Num());
        return;
    }
    
    // Check data range
    if (InImageData.Num() > 0)
    {
        float MinVal = *Algo::MinElement(InImageData);
        float MaxVal = *Algo::MaxElement(InImageData);
        UE_LOG(LogLiverAI, Warning, TEXT("Data range: %.3f to %.3f"), MinVal, MaxVal);
        
        // Check if there is valid data
        int32 NonZeroCount = 0;
        for (const float& Val : InImageData)
        {
            if (FMath::Abs(Val) > 0.001f)
            {
                NonZeroCount++;
            }
        }
        UE_LOG(LogLiverAI, Warning, TEXT("Non-zero voxels: %d / %d (%.1f%%)"), 
            NonZeroCount, InImageData.Num(), 
            (float)NonZeroCount / InImageData.Num() * 100.0f);
    }
    
    ImageData = InImageData;
    ImageWidth = Width;
    ImageHeight = Height;
    ImageDepth = Depth;
    CurrentSlice = Depth / 2;
    
    UE_LOG(LogLiverAI, Warning, TEXT("Set current slice to: %d"), CurrentSlice);
    
    CreateOrUpdateTexture();
    UpdateSliceTexture();
    UpdateSliceInfo();
    
    UE_LOG(LogLiverAI, Warning, TEXT("✅ SetImageData completed"));
}

void SLiverSliceViewer::SetMaskData(const TArray<uint8>& InMaskData, int32 Width, int32 Height, int32 Depth)
{
    UE_LOG(LogLiverAI, Warning, TEXT("=== SetMaskData called ==="));
    UE_LOG(LogLiverAI, Warning, TEXT("Mask dimensions: %d x %d x %d"), Width, Height, Depth);
    UE_LOG(LogLiverAI, Warning, TEXT("Current image dimensions: %d x %d x %d"), ImageWidth, ImageHeight, ImageDepth);
    
    if (Width != ImageWidth || Height != ImageHeight || Depth != ImageDepth)
    {
        UE_LOG(LogLiverAI, Warning, TEXT("Mask dimensions don't match image dimensions"));
        return;
    }
    
    // Count mask values
    TMap<uint8, int32> MaskCounts;
    for (const uint8& MaskVal : InMaskData)
    {
        MaskCounts.FindOrAdd(MaskVal)++;
    }
    
    UE_LOG(LogLiverAI, Warning, TEXT("Mask label counts:"));
    for (const auto& Pair : MaskCounts)
    {
        UE_LOG(LogLiverAI, Warning, TEXT("  Label %d: %d voxels"), Pair.Key, Pair.Value);
    }
    
    MaskData = InMaskData;
    UpdateSliceTexture();
    
    UE_LOG(LogLiverAI, Warning, TEXT("✅ SetMaskData completed"));
}

void SLiverSliceViewer::ClearData()
{
    UE_LOG(LogLiverAI, Warning, TEXT("=== ClearData called ==="));
    
    ImageData.Empty();
    MaskData.Empty();
    CurrentSlice = 0;
    ImageWidth = 0;
    ImageHeight = 0;
    ImageDepth = 0;
    
    if (SliceImage.IsValid())
    {
        SliceImage->SetImage(nullptr);
    }
    
    if (SliceTexture)
    {
        SliceTexture = nullptr;
    }
    
    UpdateSliceInfo();
    
    UE_LOG(LogLiverAI, Warning, TEXT("✅ ClearData completed"));
}

void SLiverSliceViewer::SetWindowLevel(float WindowCenterValue, float WindowWidthValue)
{
    WindowCenter = WindowCenterValue;
    WindowWidth = WindowWidthValue;
    UpdateSliceTexture();
}

void SLiverSliceViewer::SetMaskOpacity(float Opacity)
{
    MaskOpacity = FMath::Clamp(Opacity, 0.0f, 1.0f);
    UpdateSliceTexture();
}

void SLiverSliceViewer::SetZoomFactor(float ZoomFactorValue)
{
    ZoomFactor = FMath::Clamp(ZoomFactorValue, 0.1f, 10.0f);
    RefreshDisplay();
}

void SLiverSliceViewer::CreateOrUpdateTexture()
{
    if (ImageData.Num() == 0) 
    {
        UE_LOG(LogLiverAI, Error, TEXT("CreateOrUpdateTexture: No image data"));
        return;
    }
    
    // Calculate texture dimensions
    int32 TextureWidth, TextureHeight;
    if (ViewPlane == 0) // Axial - XY plane
    {
        TextureWidth = ImageWidth;   
        TextureHeight = ImageHeight;
    }
    else if (ViewPlane == 1) // Coronal - XZ plane
    {
        TextureWidth = ImageWidth;
        TextureHeight = ImageDepth;
    }
    else // Sagittal - YZ plane  
    {
        TextureWidth = ImageHeight;
        TextureHeight = ImageDepth;
    }
    
    UE_LOG(LogLiverAI, Warning, TEXT("CreateOrUpdateTexture: ViewPlane=%d, TextureSize=%dx%d"), 
        ViewPlane, TextureWidth, TextureHeight);
    
    // Ensure minimum size
    if (TextureWidth < 64) TextureWidth = 64;
    if (TextureHeight < 64) TextureHeight = 64;
    
    // Create or rebuild texture
    if (!SliceTexture || SliceTexture->SizeX != TextureWidth || SliceTexture->SizeY != TextureHeight)
    {
        SliceTexture = UTexture2DDynamic::Create(TextureWidth, TextureHeight, PF_B8G8R8A8);
        if (SliceTexture)
        {
            UE_LOG(LogLiverAI, Warning, TEXT("✅ Created texture: %dx%d for view plane %d"), 
                TextureWidth, TextureHeight, ViewPlane);
        }
        else
        {
            UE_LOG(LogLiverAI, Error, TEXT("❌ Failed to create texture"));
        }
    }
}

void SLiverSliceViewer::UpdateSliceTexture()
{
    if (ImageData.Num() == 0 || !SliceTexture) 
    {
        UE_LOG(LogLiverAI, Error, TEXT("UpdateSliceTexture: Missing data or texture"));
        return;
    }
    
    int32 TextureWidth = SliceTexture->SizeX;
    int32 TextureHeight = SliceTexture->SizeY;
    
    UE_LOG(LogLiverAI, Warning, TEXT("UpdateSliceTexture: Current slice %d, Texture %dx%d"), 
        CurrentSlice, TextureWidth, TextureHeight);
    
    // Validate slice index
    int32 MaxSlice = GetMaxSlices();
    if (CurrentSlice < 0 || CurrentSlice >= MaxSlice)
    {
        UE_LOG(LogLiverAI, Error, TEXT("Invalid slice index: %d (max: %d)"), CurrentSlice, MaxSlice-1);
        return;
    }
    
    // Find valid data range for normalization
    float MinValue = FLT_MAX;
    float MaxValue = -FLT_MAX;
    
    for (const float& Value : ImageData)
    {
        if (FMath::IsFinite(Value))
        {
            MinValue = FMath::Min(MinValue, Value);
            MaxValue = FMath::Max(MaxValue, Value);
        }
    }
    
    if (MaxValue <= MinValue)
    {
        MinValue = 0.0f;
        MaxValue = 1.0f;
        UE_LOG(LogLiverAI, Warning, TEXT("Using default range [0,1] due to invalid data range"));
    }
    
    UE_LOG(LogLiverAI, Warning, TEXT("Normalization range: %.3f to %.3f"), MinValue, MaxValue);
    
    // Create pixel data
    TArray<uint8> PixelData;
    PixelData.SetNum(TextureWidth * TextureHeight * 4);
    
    int32 ValidPixels = 0;
    int32 InvalidPixels = 0;
    
    // Extract current slice
    for (int32 y = 0; y < TextureHeight; y++)
    {
        for (int32 x = 0; x < TextureWidth; x++)
        {
            int32 PixelIndex = (y * TextureWidth + x) * 4;
            int32 VoxelIndex = -1;
            
            // Calculate voxel index based on view plane - assumes data is already in UE format [W,H,D]
            if (ViewPlane == 0) // Axial: XY plane, Z=CurrentSlice
            {
                if (CurrentSlice >= 0 && CurrentSlice < ImageDepth && 
                    x >= 0 && x < ImageWidth && y >= 0 && y < ImageHeight)
                {
                    // UE data format: VoxelIndex = w * (H * D) + h * D + d
                    VoxelIndex = x * (ImageHeight * ImageDepth) + y * ImageDepth + CurrentSlice;
                }
            }
            else if (ViewPlane == 1) // Coronal: XZ plane, Y=CurrentSlice
            {
                if (CurrentSlice >= 0 && CurrentSlice < ImageHeight &&
                    x >= 0 && x < ImageWidth && y >= 0 && y < ImageDepth)
                {
                    VoxelIndex = x * (ImageHeight * ImageDepth) + CurrentSlice * ImageDepth + y;
                }
            }
            else // Sagittal: YZ plane, X=CurrentSlice
            {
                if (CurrentSlice >= 0 && CurrentSlice < ImageWidth &&
                    x >= 0 && x < ImageHeight && y >= 0 && y < ImageDepth)
                {
                    VoxelIndex = CurrentSlice * (ImageHeight * ImageDepth) + x * ImageDepth + y;
                }
            }
            
            uint8 GrayValue = 0;
            if (VoxelIndex >= 0 && VoxelIndex < ImageData.Num())
            {
                float Value = ImageData[VoxelIndex];
                if (FMath::IsFinite(Value))
                {
                    float NormalizedValue = (Value - MinValue) / (MaxValue - MinValue);
                    GrayValue = (uint8)(FMath::Clamp(NormalizedValue, 0.0f, 1.0f) * 255.0f);
                    ValidPixels++;
                }
                else
                {
                    InvalidPixels++;
                }
            }
            
            uint8 R = GrayValue, G = GrayValue, B = GrayValue, A = 255;
            
            // Apply mask overlay (if available)
            if (bShowMask && MaskData.Num() > 0 && VoxelIndex >= 0 && VoxelIndex < MaskData.Num())
            {
                uint8 MaskValue = MaskData[VoxelIndex];
                if (MaskValue > 0)
                {
                    FLinearColor MaskColor = GetMaskColor(MaskValue);
                    float Alpha = MaskColor.A * MaskOpacity;
                    R = (uint8)(GrayValue * (1.0f - Alpha) + MaskColor.R * 255.0f * Alpha);
                    G = (uint8)(GrayValue * (1.0f - Alpha) + MaskColor.G * 255.0f * Alpha);
                    B = (uint8)(GrayValue * (1.0f - Alpha) + MaskColor.B * 255.0f * Alpha);
                }
            }
            
            // Store in BGRA format
            PixelData[PixelIndex + 0] = B;
            PixelData[PixelIndex + 1] = G;
            PixelData[PixelIndex + 2] = R;
            PixelData[PixelIndex + 3] = A;
        }
    }
    
    UE_LOG(LogLiverAI, Warning, TEXT("Pixel processing: %d valid, %d invalid"), ValidPixels, InvalidPixels);
    
    // Update texture on render thread
    TArray<uint8> PixelDataCopy = PixelData;
    
    ENQUEUE_RENDER_COMMAND(UpdateDynamicTextureCommand)(
        [SliceTexture = this->SliceTexture, PixelDataCopy, TextureWidth, TextureHeight](FRHICommandListImmediate& RHICmdList)
        {
            if (SliceTexture && SliceTexture->GetResource())
            {
                FTexture2DDynamicResource* TextureResource = static_cast<FTexture2DDynamicResource*>(SliceTexture->GetResource());
                if (TextureResource && TextureResource->GetTexture2DRHI())
                {
                    FUpdateTextureRegion2D Region(0, 0, 0, 0, TextureWidth, TextureHeight);
                    FTexture2DRHIRef TextureRHI = TextureResource->GetTexture2DRHI();
                    RHIUpdateTexture2D(TextureRHI, 0, Region, TextureWidth * 4, PixelDataCopy.GetData());
                }
            }
        }
    );
    
    // Update Slate brush
    if (!SliceBrush.IsValid())
    {
        SliceBrush = MakeShareable(new FSlateBrush());
    }
    SliceBrush->SetResourceObject(SliceTexture);
    SliceBrush->ImageSize = FVector2D(TextureWidth, TextureHeight);
    SliceBrush->DrawAs = ESlateBrushDrawType::Image;
    
    if (SliceImage.IsValid())
    {
        SliceImage->SetImage(SliceBrush.Get());
    }
    
    UE_LOG(LogLiverAI, Warning, TEXT("✅ UpdateSliceTexture completed"));
}

void SLiverSliceViewer::ExtractSliceData(TArray<uint8>& OutPixelData, int32 TextureWidth, int32 TextureHeight)
{
    OutPixelData.SetNum(TextureWidth * TextureHeight * 4); // BGRA8
    
    // Find min/max for proper contrast
    float MinValue = FLT_MAX;
    float MaxValue = -FLT_MAX;
    
    for (const float& Value : ImageData)
    {
        if (FMath::IsFinite(Value))
        {
            MinValue = FMath::Min(MinValue, Value);
            MaxValue = FMath::Max(MaxValue, Value);
        }
    }
    
    if (MaxValue <= MinValue)
    {
        MinValue = 0.0f;
        MaxValue = 1.0f;
    }
    
    // Extract SINGLE slice properly with UE data format
    for (int32 y = 0; y < TextureHeight; y++)
    {
        for (int32 x = 0; x < TextureWidth; x++)
        {
            int32 PixelIndex = (y * TextureWidth + x) * 4;
            int32 VoxelIndex = -1;
            
            // Calculate correct voxel index for UE data format [W,H,D]
            if (ViewPlane == 0) // Axial - XY plane, Z slice
            {
                if (CurrentSlice < ImageDepth && x < ImageWidth && y < ImageHeight)
                {
                    VoxelIndex = x * (ImageHeight * ImageDepth) + y * ImageDepth + CurrentSlice;
                }
            }
            else if (ViewPlane == 1) // Coronal - XZ plane, Y slice  
            {
                if (CurrentSlice < ImageHeight && x < ImageWidth && y < ImageDepth)
                {
                    VoxelIndex = x * (ImageHeight * ImageDepth) + CurrentSlice * ImageDepth + y;
                }
            }
            else // Sagittal - YZ plane, X slice
            {
                if (CurrentSlice < ImageWidth && x < ImageHeight && y < ImageDepth)
                {
                    VoxelIndex = CurrentSlice * (ImageHeight * ImageDepth) + x * ImageDepth + y;
                }
            }
            
            uint8 GrayValue = 0;
            if (VoxelIndex >= 0 && VoxelIndex < ImageData.Num())
            {
                float NormalizedValue = (ImageData[VoxelIndex] - MinValue) / (MaxValue - MinValue);
                GrayValue = (uint8)(FMath::Clamp(NormalizedValue, 0.0f, 1.0f) * 255.0f);
            }
            
            uint8 R = GrayValue, G = GrayValue, B = GrayValue, A = 255;
            
            // Apply mask overlay
            if (bShowMask && MaskData.Num() > 0 && VoxelIndex >= 0 && VoxelIndex < MaskData.Num())
            {
                uint8 MaskValue = MaskData[VoxelIndex];
                if (MaskValue > 0)
                {
                    FLinearColor MaskColor = GetMaskColor(MaskValue);
                    float Alpha = MaskColor.A * MaskOpacity;
                    R = (uint8)(GrayValue * (1.0f - Alpha) + MaskColor.R * 255.0f * Alpha);
                    G = (uint8)(GrayValue * (1.0f - Alpha) + MaskColor.G * 255.0f * Alpha);
                    B = (uint8)(GrayValue * (1.0f - Alpha) + MaskColor.B * 255.0f * Alpha);
                }
            }
            
            OutPixelData[PixelIndex + 0] = B;
            OutPixelData[PixelIndex + 1] = G;
            OutPixelData[PixelIndex + 2] = R;
            OutPixelData[PixelIndex + 3] = A;
        }
    }
}

uint8 SLiverSliceViewer::ApplyWindowLevel(float Intensity) const
{
    float MinValue = WindowCenter - WindowWidth * 0.5f;
    float MaxValue = WindowCenter + WindowWidth * 0.5f;
    float NormalizedIntensity = (Intensity - MinValue) / (MaxValue - MinValue);
    NormalizedIntensity = FMath::Clamp(NormalizedIntensity, 0.0f, 1.0f);
    return (uint8)(NormalizedIntensity * 255.0f);
}

int32 SLiverSliceViewer::GetVoxelIndex(int32 X, int32 Y, int32 Slice, int32 PlaneType) const
{
    // Updated for UE data format [W,H,D]
    if (PlaneType == 0) // Axial
    {
        if (Slice < ImageDepth && X < ImageWidth && Y < ImageHeight)
            return X * (ImageHeight * ImageDepth) + Y * ImageDepth + Slice;
    }
    else if (PlaneType == 1) // Coronal
    {
        if (Slice < ImageHeight && X < ImageWidth && Y < ImageDepth)
            return X * (ImageHeight * ImageDepth) + Slice * ImageDepth + Y;
    }
    else // Sagittal
    {
        if (Slice < ImageWidth && X < ImageHeight && Y < ImageDepth)
            return Slice * (ImageHeight * ImageDepth) + X * ImageDepth + Y;
    }
    return -1;
}

FLinearColor SLiverSliceViewer::GetMaskColor(uint8 ClassId) const
{
    switch (ClassId)
    {
        case 1: return FLinearColor(0.0f, 1.0f, 0.0f, 0.5f); // Liver - Green
        case 2: return FLinearColor(1.0f, 0.0f, 0.0f, 0.6f); // Vessel - Red
        case 3: return FLinearColor(1.0f, 1.0f, 0.0f, 0.7f); // Tumor - Yellow
        default: return FLinearColor(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

int32 SLiverSliceViewer::GetMaxSlices() const
{
    if (ViewPlane == 0) return ImageDepth;
    else if (ViewPlane == 1) return ImageHeight;
    else return ImageWidth;
}

int32 SLiverSliceViewer::ClampSliceIndex(int32 SliceIndex) const
{
    int32 MaxSlice = GetMaxSlices();
    return FMath::Clamp(SliceIndex, 0, MaxSlice - 1);
}

void SLiverSliceViewer::NextSlice()
{
    int32 MaxSlice = GetMaxSlices();
    if (CurrentSlice < MaxSlice - 1)
    {
        CurrentSlice++;
        UpdateSliceTexture();
        UpdateSliceInfo();
        
        // Trigger synchronization event if this is the original viewer
        OnSliceChanged.ExecuteIfBound(CurrentSlice);
    }
}

void SLiverSliceViewer::PreviousSlice()
{
    if (CurrentSlice > 0)
    {
        CurrentSlice--;
        UpdateSliceTexture();
        UpdateSliceInfo();
        
        // Trigger synchronization event if this is the original viewer
        OnSliceChanged.ExecuteIfBound(CurrentSlice);
    }
}

void SLiverSliceViewer::SetCurrentSlice(int32 SliceIndex)
{
    CurrentSlice = ClampSliceIndex(SliceIndex);
    UpdateSliceTexture();
    UpdateSliceInfo();
}

void SLiverSliceViewer::SetViewPlane(int32 Plane)
{
    if (Plane >= 0 && Plane <= 2 && Plane != ViewPlane)
    {
        ViewPlane = Plane;
        CurrentSlice = GetMaxSlices() / 2;
        CreateOrUpdateTexture();
        UpdateSliceTexture();
        UpdateSliceInfo();
    }
}

FText SLiverSliceViewer::GetSliceText() const
{
    if (ImageData.Num() == 0)
    {
        return FText::FromString(TEXT("No Data"));
    }
    
    int32 MaxSlice = GetMaxSlices();
    FString PlaneText = GetViewPlaneText().ToString();
    
    return FText::FromString(FString::Printf(TEXT("%s - Slice %d / %d"), 
        *PlaneText, CurrentSlice + 1, MaxSlice));
}

FText SLiverSliceViewer::GetViewPlaneText() const
{
    if (ViewPlane == 0) return FText::FromString(TEXT("Axial"));
    else if (ViewPlane == 1) return FText::FromString(TEXT("Coronal"));
    else return FText::FromString(TEXT("Sagittal"));
}

void SLiverSliceViewer::UpdateSliceInfo()
{
    if (SliceInfoText.IsValid())
    {
        SliceInfoText->SetText(GetSliceText());
    }
}

void SLiverSliceViewer::RefreshDisplay()
{
    UpdateSliceTexture();
    UpdateSliceInfo();
}

FReply SLiverSliceViewer::OnPreviousClicked()
{
    PreviousSlice();
    return FReply::Handled();
}

FReply SLiverSliceViewer::OnNextClicked()
{
    NextSlice();
    return FReply::Handled();
}

FReply SLiverSliceViewer::OnMouseWheel(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent)
{
    float WheelDelta = MouseEvent.GetWheelDelta();
    
    if (WheelDelta > 0)
    {
        PreviousSlice();
    }
    else if (WheelDelta < 0)
    {
        NextSlice();
    }
    
    // Notify parent widget for synchronization
    OnSliceChanged.ExecuteIfBound(CurrentSlice);
    
    return FReply::Handled();
}