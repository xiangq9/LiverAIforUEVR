#pragma once
#include "Interfaces/IFileService.h"

class LIVERIMAGEAI_API FFileService : public IFileService
{
public:
    FFileService();
    virtual ~FFileService();
    
    // IFileService interface
    virtual bool ValidateFilePath(const FString& FilePath) const override;
    virtual bool FileExists(const FString& FilePath) const override;
    virtual FString GetFileExtension(const FString& FilePath) const override;
    virtual int64 GetFileSize(const FString& FilePath) const override;
    virtual FString FormatFileSize(int64 SizeInBytes) const override;
    virtual bool OpenFileDialog(const FString& Title, const FString& FileTypes, FString& OutFilename) override;
    virtual bool SaveFileDialog(const FString& Title, const FString& FileTypes, FString& OutFilename) override;
};