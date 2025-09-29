#include "Utils/ImageDataProcessor.h"
#include "Misc/Base64.h"
#include "Misc/DateTime.h"

TArray<float> FImageDataProcessor::ConvertBase64ToFloatArray(const FString& Base64String)
{
    TArray<float> FloatArray;
    TArray<uint8> ByteArray;
    
    if (FBase64::Decode(Base64String, ByteArray))
    {
        int32 FloatCount = ByteArray.Num() / sizeof(float);
        FloatArray.SetNum(FloatCount);
        FMemory::Memcpy(FloatArray.GetData(), ByteArray.GetData(), ByteArray.Num());
    }
    
    return FloatArray;
}

TArray<uint8> FImageDataProcessor::ConvertBase64ToByteArray(const FString& Base64String)
{
    TArray<uint8> ByteArray;
    FBase64::Decode(Base64String, ByteArray);
    return ByteArray;
}

void FImageDataProcessor::NormalizeImageData(TArray<float>& ImageData)
{
    if (ImageData.Num() == 0)
    {
        return;
    }
    
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
    
    if (MaxValue > MinValue)
    {
        float Range = MaxValue - MinValue;
        for (float& Value : ImageData)
        {
            if (FMath::IsFinite(Value))
            {
                Value = (Value - MinValue) / Range;
            }
        }
    }
}

FLinearColor FImageDataProcessor::GetOrganColor(const FString& OrganName)
{
    FString LowerName = OrganName.ToLower();
    
    if (LowerName.Contains(TEXT("liver")))
        return FLinearColor(0.0f, 0.8f, 0.0f, 0.8f);
    else if (LowerName.Contains(TEXT("vessel")))
        return FLinearColor(0.8f, 0.0f, 0.0f, 0.8f);
    else if (LowerName.Contains(TEXT("tumor")))
        return FLinearColor(0.8f, 0.8f, 0.0f, 0.9f);
    else
        return FLinearColor(0.7f, 0.7f, 0.7f, 0.8f);
}

FString FImageDataProcessor::GenerateRequestId()
{
    FDateTime Now = FDateTime::Now();
    return FString::Printf(TEXT("ue_request_%s"), *Now.ToString(TEXT("%Y%m%d_%H%M%S")));
}