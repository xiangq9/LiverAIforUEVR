#include "UI/SliceViewerWidget.h"
#include "SLiverSliceViewer.h"  // 使用原有的 SLiverSliceViewer
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Text/STextBlock.h"
#include "Styling/CoreStyle.h"

void SSliceViewerWidget::Construct(const FArguments& InArgs)
{
    Title = InArgs._Title;
    
    ChildSlot
    [
        SNew(SBorder)
        .BorderImage(FCoreStyle::Get().GetBrush("ToolPanel.GroupBorder"))
        [
            SNew(SVerticalBox)
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5)
            [
                SAssignNew(TitleText, STextBlock)
                .Text(FText::FromString(Title))
                .Font(FCoreStyle::GetDefaultFontStyle("Bold", 11))
                .Justification(ETextJustify::Center)
            ]
            + SVerticalBox::Slot()
            .FillHeight(1.0f)
            .HAlign(HAlign_Center)
            .VAlign(VAlign_Center)
            [
                SNew(SBox)
                .WidthOverride(500.0f)
                .HeightOverride(500.0f)
                [
                    SAssignNew(InternalSliceViewer, SLiverSliceViewer)
                ]
            ]
        ]
    ];
    
    // 设置事件转发
    if (InternalSliceViewer.IsValid())
    {
        InternalSliceViewer->OnSliceChanged.BindLambda([this](int32 NewSlice)
        {
            OnSliceChanged.ExecuteIfBound(NewSlice);
        });
    }
}

void SSliceViewerWidget::SetImageData(const TArray<float>& ImageData, int32 Width, int32 Height, int32 Depth)
{
    if (InternalSliceViewer.IsValid())
    {
        InternalSliceViewer->SetImageData(ImageData, Width, Height, Depth);
    }
}

void SSliceViewerWidget::SetMaskData(const TArray<uint8>& MaskData, int32 Width, int32 Height, int32 Depth)
{
    if (InternalSliceViewer.IsValid())
    {
        InternalSliceViewer->SetMaskData(MaskData, Width, Height, Depth);
    }
}

void SSliceViewerWidget::ClearData()
{
    if (InternalSliceViewer.IsValid())
    {
        InternalSliceViewer->ClearData();
    }
}

void SSliceViewerWidget::SetViewPlane(int32 Plane)
{
    if (InternalSliceViewer.IsValid())
    {
        InternalSliceViewer->SetViewPlane(Plane);
    }
}

void SSliceViewerWidget::SetCurrentSlice(int32 Slice)
{
    if (InternalSliceViewer.IsValid())
    {
        InternalSliceViewer->SetCurrentSlice(Slice);
    }
}

int32 SSliceViewerWidget::GetCurrentSlice() const
{
    if (InternalSliceViewer.IsValid())
    {
        return InternalSliceViewer->GetCurrentSlice();
    }
    return 0;
}

void SSliceViewerWidget::SetWindowLevel(float WindowCenter, float WindowWidth)
{
    if (InternalSliceViewer.IsValid())
    {
        InternalSliceViewer->SetWindowLevel(WindowCenter, WindowWidth);
    }
}

void SSliceViewerWidget::SetMaskOpacity(float Opacity)
{
    if (InternalSliceViewer.IsValid())
    {
        InternalSliceViewer->SetMaskOpacity(Opacity);
    }
}

void SSliceViewerWidget::SetZoomFactor(float ZoomFactor)
{
    if (InternalSliceViewer.IsValid())
    {
        InternalSliceViewer->SetZoomFactor(ZoomFactor);
    }
}

void SSliceViewerWidget::NextSlice()
{
    if (InternalSliceViewer.IsValid())
    {
        InternalSliceViewer->NextSlice();
    }
}

void SSliceViewerWidget::PreviousSlice()
{
    if (InternalSliceViewer.IsValid())
    {
        InternalSliceViewer->PreviousSlice();
    }
}

int32 SSliceViewerWidget::GetMaxSlices() const
{
    if (InternalSliceViewer.IsValid())
    {
        return InternalSliceViewer->GetMaxSlices();
    }
    return 1;
}

int32 SSliceViewerWidget::GetViewPlane() const
{
    if (InternalSliceViewer.IsValid())
    {
        return InternalSliceViewer->GetViewPlane();
    }
    return 0;
}