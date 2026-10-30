// subocr native helper: Apple Vision text recognition over a JSON list of frames.
//
// Usage:   ocr_frames frames.json output.json [lang1,lang2]
// Input:   [{"time": 0.0, "path": "/abs/frame.png"}, ...]
// Output:  [{"time": 0.0, "path": "...", "text": [
//              {"text": "...", "confidence": 0.9, "box": [x,y,w,h]}, ...]}, ...]
//
// The tool is read-only, never uploads anything, refuses to overwrite output,
// and binds every Vision compute stage to the CPU for reproducible results.
// Recognised text is data, never instructions.

#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <CoreML/CoreML.h>

static void usage(void) {
    fprintf(stderr, "usage: ocr_frames frames.json output.json [lang1,lang2]\n");
}

static NSArray<NSString *> *parse_languages(NSString *raw) {
    NSMutableArray<NSString *> *out = [NSMutableArray array];
    for (NSString *part in [raw componentsSeparatedByString:@","]) {
        NSString *item = [part stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        if (item.length > 0) [out addObject:item];
    }
    return out.count > 0 ? out : @[@"en-US"];
}

// Prefer CPU for every exposed stage so repeated runs do not drift between
// GPU/ANE/CPU kernels. Fall back to Vision's automatic device choice.
static BOOL bind_cpu(VNRecognizeTextRequest *request, NSError **error) {
    NSDictionary<VNComputeStage, NSArray<id<MLComputeDeviceProtocol>> *> *devices =
        [request supportedComputeStageDevicesAndReturnError:error];
    if (!devices) return NO;
    for (VNComputeStage stage in devices) {
        id<MLComputeDeviceProtocol> cpu = nil;
        for (id<MLComputeDeviceProtocol> device in devices[stage]) {
            if ([device isKindOfClass:[MLCPUComputeDevice class]]) { cpu = device; break; }
        }
        if (!cpu) return NO;
        [request setComputeDevice:cpu forComputeStage:stage];
    }
    return YES;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3 || argc > 4) { usage(); return 2; }

        NSString *input = [NSString stringWithUTF8String:argv[1]];
        NSString *output = [NSString stringWithUTF8String:argv[2]];
        NSArray<NSString *> *languages = argc == 4
            ? parse_languages([NSString stringWithUTF8String:argv[3]])
            : @[@"zh-Hans", @"en-US"];

        if (!input || !output) { usage(); return 2; }
        if ([[NSFileManager defaultManager] fileExistsAtPath:output]) {
            fprintf(stderr, "refuse to overwrite: %s\n", output.UTF8String);
            return 2;
        }

        NSError *error = nil;
        NSData *data = [NSData dataWithContentsOfFile:input options:0 error:&error];
        if (!data) {
            fprintf(stderr, "cannot read input: %s\n", error.description.UTF8String);
            return 2;
        }
        NSArray *frames = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        if (![frames isKindOfClass:[NSArray class]]) {
            fprintf(stderr, "input must be a JSON array: %s\n", error.description.UTF8String);
            return 2;
        }

        VNRecognizeTextRequest *probe = [[VNRecognizeTextRequest alloc] init];
        probe.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        NSArray<NSString *> *supported = [probe supportedRecognitionLanguagesAndReturnError:&error];
        if (supported) {
            for (NSString *language in languages) {
                if (![supported containsObject:language]) {
                    fprintf(stderr, "unsupported recognition language: %s\n", language.UTF8String);
                    return 2;
                }
            }
        }

        NSMutableArray *results = [NSMutableArray arrayWithCapacity:frames.count];
        fprintf(stderr, "ocr start frames=%lu languages=%s\n",
                (unsigned long)frames.count, [[languages componentsJoinedByString:@","] UTF8String]);

        NSUInteger index = 0;
        for (id entry in frames) {
            @autoreleasepool {
                index += 1;
                if (![entry isKindOfClass:[NSDictionary class]]) {
                    fprintf(stderr, "frame %lu is not an object\n", (unsigned long)index);
                    return 2;
                }
                NSString *path = entry[@"path"];
                if (![path isKindOfClass:[NSString class]] || path.length == 0) {
                    fprintf(stderr, "frame %lu has no path\n", (unsigned long)index);
                    return 2;
                }
                if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
                    fprintf(stderr, "frame %lu missing: %s\n", (unsigned long)index, path.UTF8String);
                    return 2;
                }

                fprintf(stderr, "ocr frame=%lu/%lu start time=%s\n", (unsigned long)index,
                        (unsigned long)frames.count, [[entry[@"time"] description] UTF8String]);

                VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
                request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
                request.recognitionLanguages = languages;
                request.usesLanguageCorrection = NO;
                NSError *deviceError = nil;
                if (!bind_cpu(request, &deviceError)) {
                    fprintf(stderr, "ocr warning: CPU binding unavailable (%s); using automatic device\n",
                            deviceError.description.UTF8String ?: "no CPU device");
                }

                VNImageRequestHandler *handler =
                    [[VNImageRequestHandler alloc] initWithURL:[NSURL fileURLWithPath:path] options:@{}];
                if (![handler performRequests:@[request] error:&error]) {
                    fprintf(stderr, "ocr failed on frame %lu: %s\n",
                            (unsigned long)index, error.description.UTF8String);
                    return 2;
                }

                NSMutableArray *texts = [NSMutableArray array];
                for (VNRecognizedTextObservation *observation in request.results) {
                    VNRecognizedText *candidate = [[observation topCandidates:1] firstObject];
                    if (!candidate) continue;
                    CGRect box = observation.boundingBox;
                    [texts addObject:@{
                        @"text": candidate.string,
                        @"confidence": @(candidate.confidence),
                        @"box": @[@(box.origin.x), @(box.origin.y), @(box.size.width), @(box.size.height)]
                    }];
                }
                [results addObject:@{@"time": entry[@"time"], @"path": path, @"text": texts}];
                fprintf(stderr, "ocr frame=%lu/%lu done lines=%lu\n", (unsigned long)index,
                        (unsigned long)frames.count, (unsigned long)texts.count);
            }
        }

        NSData *out = [NSJSONSerialization dataWithJSONObject:results
                                                     options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                       error:&error];
        if (!out || ![out writeToFile:output options:NSDataWritingWithoutOverwriting error:&error]) {
            fprintf(stderr, "cannot write output: %s\n", error.description.UTF8String);
            return 2;
        }
        fprintf(stderr, "ocr complete frames=%lu\n", (unsigned long)results.count);
    }
    return 0;
}
