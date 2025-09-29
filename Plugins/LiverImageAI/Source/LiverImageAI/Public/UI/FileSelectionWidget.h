#pragma once
#include "Widgets/SCompoundWidget.h"

class IFileService;

class LIVERIMAGEAI_API SFileSelectionWidget : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SFileSelectionWidget) {}
        SLATE_ARGUMENT(FString, Label)
        SLATE_ARGUMENT(FString, FileTypes)
        SLATE_ARGUMENT(FString, HintText)
    SLATE_END_ARGS()
    
    void Construct(const FArguments& InArgs);
    
    FString GetFilePath() const;
    void SetFilePath(const FString& Path);
    void ClearPath();
    bool IsPathValid() const;
    
    DECLARE_DELEGATE_OneParam(FOnFileSelected, const FString&);
    FOnFileSelected OnFileSelected;
    
private:
    TSharedPtr<IFileService> FileService;
    TSharedPtr<SEditableTextBox> FilePathText;
    TSharedPtr<STextBlock> FileSizeText;
    
    FString Label;
    FString FileTypes;
    FString CurrentPath;
    
    FReply OnBrowseClicked();
    void UpdateFileInfo();
};