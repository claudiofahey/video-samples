/*
 * Copyright (c) 2019 Dell Inc., or its subsidiaries. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 */
package io.pravega.example.videoprocessor;

import io.pravega.client.stream.StreamCut;
import io.pravega.connectors.flink.FlinkPravegaReader;
import io.pravega.connectors.flink.FlinkPravegaWriter;
import io.pravega.connectors.flink.PravegaWriterMode;
import io.pravega.example.flinkprocessor.AbstractJob;
import io.pravega.example.common.ChunkedVideoFrame;
import io.pravega.example.common.VideoFrame;
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.streaming.api.functions.timestamps.BoundedOutOfOrdernessTimestampExtractor;
import org.apache.flink.streaming.api.functions.windowing.ProcessWindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Timestamp;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.lang.Math.max;

/**
 * A Flink job that reads images from multiple cameras stored in a Pravega stream, combines them
 * into a square grid of images (like a security camera monitor), and writes the resulting
 * images to another Pravega stream.
 * Images are chunked into 512 KB chunks to allow for very large images.
 */
public class MultiVideoGridJob extends AbstractJob {
    private static Logger log = LoggerFactory.getLogger(MultiVideoGridJob.class);

    /**
     * The entry point for Flink applications.
     *
     * @param args Command line arguments
     */
    public static void main(String... args) {
        VideoAppConfiguration config = new VideoAppConfiguration(args);
        log.info("config: {}", config);
        MultiVideoGridJob job = new MultiVideoGridJob(config);
        job.run();
    }

    public MultiVideoGridJob(VideoAppConfiguration config) {
        super(config);
    }

    @Override
    public VideoAppConfiguration getConfig() {
        return (VideoAppConfiguration) super.getConfig();
    }

    public void run() {
        try {
            final String jobName = MultiVideoGridJob.class.getName();
            final StreamExecutionEnvironment env = initializeFlinkStreaming();
            createStream(getConfig().getInputStreamConfig());
            createStream(getConfig().getOutputStreamConfig());

            final StreamCut startStreamCut;
            if (getConfig().isStartAtTail()) {
                startStreamCut = getStreamInfo(getConfig().getInputStreamConfig().getStream()).getTailStreamCut();
            } else {
                startStreamCut = StreamCut.UNBOUNDED;
            }

            // Read chunked video frames from Pravega.
            // Operator: input-source
            // parallelism: min of # of segments, getReaderParallelism()
            final FlinkPravegaReader<ChunkedVideoFrame> flinkPravegaReader = FlinkPravegaReader.<ChunkedVideoFrame>builder()
                    .withPravegaConfig(getConfig().getPravegaConfig())
                    .forStream(getConfig().getInputStreamConfig().getStream(), startStreamCut, StreamCut.UNBOUNDED)
                    .withDeserializationSchema(new ChunkedVideoFrameDeserializationSchema())
                    .build();
            final DataStream<ChunkedVideoFrame> inChunkedVideoFrames = env
                    .addSource(flinkPravegaReader)
                    .setParallelism(getConfig().getReaderParallelism())
                    .uid("input-source")
                    .name("input-source");
            inChunkedVideoFrames.printToErr();

            // Assign timestamps and watermarks based on timestamp in each chunk.
            // Operator: assignTimestampsAndWatermarks
            // parallelism: min of # of segments, getReaderParallelism()
            final DataStream<ChunkedVideoFrame> inChunkedVideoFramesWithTimestamps = inChunkedVideoFrames
                    .assignTimestampsAndWatermarks(
                        new BoundedOutOfOrdernessTimestampExtractor<ChunkedVideoFrame>(
                                Time.milliseconds(getConfig().getMaxOutOfOrdernessMs())) {
                            @Override
                            public long extractTimestamp(ChunkedVideoFrame element) {
                                return element.timestamp.getTime();
                            }
                        })
                    .uid("assignTimestampsAndWatermarks")
                    .name("assignTimestampsAndWatermarks");
//            inChunkedVideoFramesWithTimestamps.printToErr().uid("inChunkedVideoFramesWithTimestamps-print").name("inChunkedVideoFramesWithTimestamps-print");

            // Unchunk (disabled).
            // Operator: ChunkedVideoFrameReassembler
            // parallelism: min of # of segments, getReaderParallelism()
            final DataStream<VideoFrame> inVideoFrames = inChunkedVideoFramesWithTimestamps
                    .map(VideoFrame::new)
                    .uid("ChunkedVideoFrameReassembler")
                    .name("ChunkedVideoFrameReassembler");

            // For each camera and window, get the most recent frame.
            // Operator: lastVideoFramePerCamera
            // input parallelism: min of # of segments, getReaderParallelism()
            // output parallelism: # of cameras
            final long periodMs = (long) (1000.0 / getConfig().getFramesPerSec());
            final int imageWidth = getConfig().getImageWidth();
            final int imageHeight = getConfig().getImageHeight();
            final DataStream<VideoFrame> lastVideoFramePerCamera = inVideoFrames
                    .keyBy((KeySelector<VideoFrame, Integer>) value -> value.camera)
                    .window(TumblingEventTimeWindows.of(Time.milliseconds(periodMs)))
                    .maxBy("timestamp")
                    .uid("lastVideoFramePerCamera")
                    .name("lastVideoFramePerCamera");

            // Rebalance (round-robin) to all task slots.
            // input parallelism:  # of cameras
            // output parallelism: default parallelism
            final DataStream<VideoFrame> rebalancedVideoFrames;
            if (getConfig().isEnableRebalance()) {
                rebalancedVideoFrames = lastVideoFramePerCamera.rebalance();
            } else {
                rebalancedVideoFrames = lastVideoFramePerCamera;
            }

            // Resize all input images. Frames within each input camera will be processed sequentially.
            // Operator: ImageResizer
            // parallelism: # of cameras
            lastVideoFramePerCamera.printToErr();
            final DataStream<VideoFrame> resizedVideoFrames = rebalancedVideoFrames
                    .map(frame -> {
                        ImageResizer resizer = new ImageResizer(imageWidth, imageHeight);
                        frame.data = resizer.resize(frame.data);
                        frame.hash = null;
                        return frame;
                    })
                    .uid("ImageResizer")
                    .name("ImageResizer");
//            resizedVideoFrames.printToErr().uid("resizedVideoFrames-print").name("resizedVideoFrames-print");;

            // Aggregate resized images. This only adds images to a hash map. The actual work is done later to increase parallelism.
            // Operator: ImageAggregator
            // input parallelism: # of cameras
            // output parallelism: # of monitors
            final DataStream<ImageAggregatorResult> aggResults = resizedVideoFrames
                    .keyBy((KeySelector<VideoFrame, Integer>) value -> getMonitorFromCamera(value.camera))
                    .window(TumblingEventTimeWindows.of(Time.milliseconds(periodMs)))
                    .process(new ProcessWindowFunction<VideoFrame, ImageAggregatorResult, Integer, TimeWindow>() {
                         @Override
                         public void process(Integer camera, Context context, Iterable<VideoFrame> elements, Collector<ImageAggregatorResult> out) throws Exception {
                             final ImageAggregatorResult result = new ImageAggregatorResult();
                             elements.forEach(element -> {
                                 result.camera = camera;
                                 result.timestamp = new Timestamp(max(result.timestamp.getTime(), element.timestamp.getTime()));
                                 result.videoFrames.put(element.camera, element);
                             });
                             out.collect(result);
                         }
                     })
                    .uid("ImageAggregator")
                    .name("ImageAggregator");

            // Assign sequential index.
            // Operator: assignSequentialIndex
            // parallelism: # of monitors
            final DataStream<OrderedImageAggregatorResult> assignSequentialIndex = aggResults
                    .keyBy((KeySelector<ImageAggregatorResult, Integer>) value -> value.camera)
                    .process(new KeyedProcessFunction<Integer, ImageAggregatorResult, OrderedImageAggregatorResult>() {
                        private ValueState<Long> indexState;

                        @Override
                        public void open(Configuration parameters) throws Exception {
                            indexState = getRuntimeContext().getState(new ValueStateDescriptor<>("index", Long.class));
                        }

                        @Override
                        public void processElement(ImageAggregatorResult value, Context ctx, Collector<OrderedImageAggregatorResult> out) throws Exception {
                            final long index = Optional.ofNullable(indexState.value()).orElse(0L);
                            out.collect(new OrderedImageAggregatorResult(index, value));
                            indexState.update(index + 1);
                        }
                    })
                    .uid("assignSequentialIndex")
                    .name("assignSequentialIndex");

            final int parallelism = env.getParallelism();
            final List<OutputTag<OrderedImageAggregatorResult>> outputTags = IntStream.range(0, parallelism).boxed()
                    .map(i -> new OutputTag<>(i.toString(), TypeInformation.of(OrderedImageAggregatorResult.class))).collect(Collectors.toList());

            final SingleOutputStreamOperator<Integer> splitStream = assignSequentialIndex
                    .process(new ProcessFunction<OrderedImageAggregatorResult, Integer>() {
                        @Override
                        public void processElement(OrderedImageAggregatorResult value, Context ctx, Collector<Integer> out) throws Exception {
                            ctx.output(outputTags.get((int) value.index % parallelism), value);
                        }
                    })
                    .uid("splitStream")
                    .name("splitStream");

            final List<DataStream<OrderedImageAggregatorResult>> splits = outputTags.stream()
                    .map(splitStream::getSideOutput).collect(Collectors.toList());

            final int ssrc = new Random().nextInt();
            final List<SingleOutputStreamOperator<OrderedVideoFrame>> orderedVideoFrames = IntStream.range(0, parallelism).boxed()
                    .map(i ->
                        splits.get(i).map(aggResult ->
                                new OrderedVideoFrame(aggResult.index, aggResult.value.asVideoFrame(imageWidth, imageHeight, ssrc, (int) aggResult.index))
                        )
                        .uid("asVideoFrame-" + i)
                        .name("asVideoFrame-" + i)
                    )
                    .collect(Collectors.toList());

            DataStream<OrderedVideoFrame> reduced = orderedVideoFrames.get(0);
            for (int operator = 0 ; operator < parallelism - 1 ; operator++) {
                final KeyedStream<OrderedVideoFrame, Integer> keyed1 = reduced
                        .keyBy((KeySelector<OrderedVideoFrame, Integer>) value -> value.value.camera);
                final KeyedStream<OrderedVideoFrame, Integer> keyed2 = orderedVideoFrames.get(operator+1)
                        .keyBy((KeySelector<OrderedVideoFrame, Integer>) value -> value.value.camera);
                reduced = keyed1
                        .connect(keyed2)
                        .process(new OrderedVideoFrameCoProcessFunction(operator, parallelism))
                        .uid("reduce-" + operator)
                        .name("reduce-" + operator);
            }

//            reduced.printToErr();

            final DataStream<VideoFrame> outVideoFrames = reduced
                    .map(x -> x.value)
                    .uid("outVideoFrames")
                    .name("outVideoFrames");

            outVideoFrames.printToErr().uid("outVideoFrames-print").name("outVideoFrames-print");

            // Split output video frames into chunks of 8 MiB or less.
            final DataStream<ChunkedVideoFrame> outChunkedVideoFrames = outVideoFrames
                    .flatMap(new VideoFrameChunker(getConfig().getChunkSizeBytes()))
                    .uid("VideoFrameChunker")
                    .name("VideoFrameChunker");
//            outChunkedVideoFrames.printToErr().setParallelism(1).uid("outChunkedVideoFrames-print").name("outChunkedVideoFrames-print");

            // Write chunks to Pravega encoded as JSON.
            final FlinkPravegaWriter<ChunkedVideoFrame> flinkPravegaWriter = FlinkPravegaWriter.<ChunkedVideoFrame>builder()
                    .withPravegaConfig(getConfig().getPravegaConfig())
                    .forStream(getConfig().getOutputStreamConfig().getStream())
                    .withSerializationSchema(new ChunkedVideoFrameSerializationSchema())
                    .withEventRouter(frame -> String.format("%d", frame.camera))
                    .withWriterMode(PravegaWriterMode.ATLEAST_ONCE)
                    .build();
            outChunkedVideoFrames
                    .addSink(flinkPravegaWriter)
                    .uid("output-sink")
                    .name("output-sink");

            log.info("Executing {} job", jobName);
            env.execute(jobName);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    static int camerasPerMonitor = 2;

    static int getMonitorFromCamera(final int camera) {
        return camera / camerasPerMonitor;
    }

    static int getPositionInMonitorFromCamera(final int camera) {
        return camera % camerasPerMonitor;
    }

    public static class ImageAggregatorAccum {
        public int camera;
        // Map from camera to last VideoFrame.
        public Map<Integer, VideoFrame> videoFrames = new HashMap<>();
        // Maximum timestamp from cameras.
        public Timestamp timestamp = new Timestamp(0);

        public ImageAggregatorAccum() {
        }

//        public ImageAggregatorAccum(ImageAggregatorAccum accum) {
//            this.camera = accum.camera;
//            this.videoFrames = accum.videoFrames;
//            this.timestamp = accum.timestamp;
//        }

        @Override
        public String toString() {
            return "ImageAggregatorAccum{" +
                    "camera=" + camera +
                    ",timestamp=" + timestamp +
                    '}';
        }
    }

    public static class ImageAggregatorResult extends ImageAggregatorAccum {
        public ImageAggregatorResult() {
        }

//        public ImageAggregatorResult(ImageAggregatorAccum accum) {
//            super(accum);
//        }
//
        public VideoFrame asVideoFrame(int imageWidth, int imageHeight, int ssrc, int frameNumber) {
            log.info("asVideoFrame: BEGIN: frameNumber={}", frameNumber);
            final ImageGridBuilder builder = new ImageGridBuilder(imageWidth, imageHeight, videoFrames.size());
            videoFrames.forEach((camera, frame) -> builder.addImage(getPositionInMonitorFromCamera(camera), frame.data));
            final long t0 = System.currentTimeMillis();
            VideoFrame videoFrame = new VideoFrame();
            videoFrame.camera = camera;
            videoFrame.ssrc = ssrc + camera;
            videoFrame.timestamp = timestamp;
            videoFrame.frameNumber = frameNumber;
            videoFrame.data = builder.getOutputImageBytes("jpg");
            videoFrame.hash = videoFrame.calculateHash();
            videoFrame.tags = new HashMap<>();
            videoFrame.tags.put("numCameras", Integer.toString(videoFrames.size()));
            videoFrame.tags.put("cameras", videoFrames.keySet().toString());

//            long sleepTime = new Random().nextInt(1000);
//            // long sleepTime = (frameNumber % 2 == 0)
//            try {
//                Thread.sleep(sleepTime);
//            } catch (InterruptedException e) {
//                throw new RuntimeException(e);
//            }

            log.info("asVideoFrame: END: frameNumber={}, TIME={}", frameNumber, System.currentTimeMillis() - t0);
            log.trace("asVideoFrame: videoFrame={}", videoFrame);
            return videoFrame;
        }
    }

    public static class OrderedImageAggregatorResult {
        final public long index;
        final public ImageAggregatorResult value;

        public OrderedImageAggregatorResult(long index, ImageAggregatorResult value) {
            this.index = index;
            this.value = value;
        }

        @Override
        public String toString() {
            return "OrderedImageAggregatorResult{" +
                    "index=" + index +
                    ", value=" + value +
                    '}';
        }
    }
}
