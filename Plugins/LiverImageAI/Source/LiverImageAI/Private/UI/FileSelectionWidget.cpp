#include "UI/FileSelectionWidget.h"
#include "Services/FileService.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/SBoxPanel.h"
#include "Styling/CoreStyle.h"

void SFileSelectionWidget::Construct(const FArguments& InArgs)
{
    Label = InArgs._Label;
    FileTypes = InArgs._FileTypes;
    FileService = MakeShareable(new FFileService());
    
    ChildSlot
    [
        SNew(SHorizontalBox)
        + SHorizontalBox::Slot()
        .FillWidth(0.3f)
        .VAlign(VAlign_Center)
        [
            SNew(STextBlock)
            .Text(FText::FromString(Label))
        ]
        + SHorizontalBox::Slot()
        .FillWidth(0.5f)
        [
            SAssignNew(FilePathText, SEditableTextBox)
            .HintText(FText::FromString(InArgs._HintText))
            .IsReadOnly(true)
        ]
        + SHorizontalBox::Slot()
        .FillWidth(0.1f)
        .Padding(5, 0)
        [
            SAssignNew(FileSizeText, STextBlock)
            .Text(FText::GetEmpty())
            .Font(FCoreStyle::GetDefaultFontStyle("Regular", 8))
        ]
        + SHorizontalBox::Slot()
        .FillWidth(0.1f)
        [
            SNew(SButton)
            .Text(FText::FromString(TEXT("Browse")))
            .OnClicked(this, &SFileSelectionWidget::OnBrowseClicked)
        ]
    ];
}

FReply SFileSelectionWidget::OnBrowseClicked()
{
    FString SelectedFile;
    if (FileService->OpenFileDialog(Label, FileTypes, SelectedFile))
    {
        SetFilePath(SelectedFile);
        OnFileSelected.ExecuteIfBound(SelectedFile);
    }
    return FReply::Handled();
}

FString SFileSelectionWidget::GetFilePath() const
{
    return CurrentPath;
}

void SFileSelectionWidget::SetFilePath(const FString& Path)
{
    CurrentPath = Path;
    if (FilePathText.IsValid())
    {
        FilePathText->SetText(FText::FromString(Path));
    }
    UpdateFileInfo();
}

void SFileSelectionWidget::ClearPath()
{
    SetFilePath(TEXT(""));
}

bool SFileSelectionWidget::IsPathValid() const
{
    return FileService->ValidateFilePath(CurrentPath);
}

void SFileSelectionWidget::UpdateFileInfo()
{
    if (FileSizeText.IsValid())
    {
        if (IsPathValid())
        {
            int64 FileSize = FileService->GetFileSize(CurrentPath);
            FString SizeStr = FileService->FormatFileSize(FileSize);
            FileSizeText->SetText(FText::FromString(SizeStr));
        }
        else
        {
            FileSizeText->SetText(FText::GetEmpty());
        }
    }
}