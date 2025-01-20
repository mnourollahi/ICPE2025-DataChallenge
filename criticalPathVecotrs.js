loadModule("/TraceCompass/Trace");
loadModule('/TraceCompass/Analysis');
loadModule('/System/Resources');
loadModule('/TraceCompass/View');
loadModule('/TraceCompass/TraceUI');

const critPathPath = "/home/masoum/output/critPaths";
const vectorPath= "/home/masoum/output/vectors";

var followName = "elasticsearch";
var numTraces = 1;

print("start");
var d = new Date();
var t = d.getTime();

var statusVectorOut = createFile(vectorPath + "Status" + t + ".csv");
var statusDurVectorOut = createFile(vectorPath + "StatusDur" + t + ".csv");
var transitionRateOut = createFile(vectorPath + "TransitionRate" + t + ".csv");
var varianceDurOut = createFile(vectorPath + "VarianceDur" + t + ".csv");
														

var header = "ThreadID,WindowStartTime(s),";
for (var i = 0; i < 13; i++) {
    header += i + ",";
}
header += "Transitions,Variance";

var statusVectorHandle = writeLine(statusVectorOut, header);
var statusDurVectorHandle = writeLine(statusDurVectorOut, header);
var transitionRateHandle = writeLine(transitionRateOut, "ThreadID,WindowStartTime(s),TransitionRate");
var varianceDurHandle = writeLine(varianceDurOut, "ThreadID,WindowStartTime(s),Variance");


for(traceNum = 1; traceNum <= numTraces; traceNum++) { 
    print("Analyzing trace " + traceNum);

    // Normally you might open the trace via openTrace, but here we just getActiveTrace:
    var trace = getActiveTrace();
    if (trace == null) {
        print("Trace is null");
    } else {
       
        getData(trace);

    }
}

// -------------------------
// getData: Processes the trace
// -------------------------
function getData(trace) {
    var events = getEventIterator(trace);

    // 1) Set up and schedule the main OS Execution Graph analysis once
    var analysis = getTraceAnalysis(trace, 'OS Execution Graph');
    analysis.schedule();
    analysis.waitForCompletion();

    var osGraph = analysis.getGraph();
    var workers = osGraph.getWorkers();
    var iter = workers.iterator();

    // 2) Batching
    var batch = [];
    var batchSize = 5; // pick a sensible number
    var batchIndex = 0;

    while (iter.hasNext()) {
        batch.push(iter.next());

        // Once the batch is full or we are at the end
        if (batch.length >= batchSize || !iter.hasNext()) {
            print("Processing batch " + batchIndex + " of size " + batch.length);
            try {
                processBatch(batch, trace, analysis);
            } catch (e) {
                print("Error processing batch " + batchIndex + ": " + e.message);
            }
            // Clear batch
            batch = [];
            batchIndex++;

            // Suggest garbage collection after each batch
            java.lang.System.gc();
        }
    }
}

function processBatch(batch, trace, analysis) {
    batch.forEach(function(worker) {
        var name = worker.getName();
        if (name.contains("elasticsearch")) {
            var info = worker.getWorkerInformation();
            var tid = info.get('TID');
            processWorkerCriticalPath(worker, trace, analysis, tid);
        }
    });
}


function processWorkerCriticalPath(worker, trace, analysis, threadid) {

    // Dispatch thread selection for TID
    var threadSignal = new org.eclipse.tracecompass.analysis.os.linux.core.signals
                          .TmfThreadSelectedSignal(this, threadid, trace);
    org.eclipse.tracecompass.tmf.core.signal.TmfSignalManager.dispatchSignal(threadSignal);

    // Create only the Critical Path module, reuse the same OS Execution Graph 'analysis'
    var critPathMod = new org.eclipse.tracecompass.analysis.graph.core.criticalpath
                          .CriticalPathModule(analysis);
    critPathMod.setTrace(trace);
    critPathMod.setParameter("workerid", worker);
    critPathMod.schedule();
    critPathMod.waitForCompletion();

    var critPath = critPathMod.getCriticalPath();
    if (!critPath) {
        print("No critical path for worker " + worker.getName());
        return;
    }

								
    var next = critPath.getHead();
    var windowSize = 100 * 1e9; // 10 seconds in nanoseconds
    var windowStart = next.getTs();
    var windowEnd = windowStart + windowSize;

    						
    var edges = org.eclipse.tracecompass.analysis.graph.core.base.TmfVertex.EdgeDirection.values();
    var statusVector = [];
    var statusDurVector = [];
    for(var x = 0; x < 15; x++){
                statusVector[x] = 0;
                statusDurVector[x] = 0;
            }
    var transitionCount = 0;
    var previousStatus = -1;
    var durationList = [];

    while (next != null) {
        var vertex = next;
        var sTime = vertex.getTs();

        if (sTime >= windowEnd) {
            var variance = calculateVariance(durationList);
            writeLine(statusVectorHandle, threadid + "," + (windowStart / 1e9) + "," + vectorToString(statusVector));																		
            writeLine(statusDurVectorHandle, threadid + "," + (windowStart / 1e9) + "," + vectorToString(statusDurVector));
            writeLine(transitionRateHandle, threadid + "," + (windowStart / 1e9) + "," + transitionCount);
            writeLine(varianceDurHandle, threadid + "," + (windowStart / 1e9) + "," + variance);

            windowStart = windowEnd;
            windowEnd += windowSize;
			
            for(var x = 0; x < 15; x++){
                statusVector[x] = 0;
                statusDurVector[x] = 0;
            }
			
			transitionCount = 0;
            previousStatus = -1;
            durationList = [];
        }		
                
       var edge = vertex.getEdge(edges[2]);
       if (!edge) {
           edge = vertex.getEdge(edges[0]);  // Fallback to vertical edge if horizontal is missing
       }
        if (edge) {
            next = vertex.getNeighborFromEdge(edge, edges[2]);
            var eTime = next.getTs();
            var elapsed = eTime - sTime;
            var status = getChar(edge.getType());
            statusVector[status]++;
            statusDurVector[status] += elapsed;
            durationList.push(elapsed);

            if (previousStatus !== -1 && previousStatus !== status) {
                transitionCount++;
            }
            previousStatus = status;
        } else {
            next = null;
        }
    }
    var closeSignal = new org.eclipse.tracecompass.tmf.core.signal.TmfTraceClosedSignal(this, trace);
    org.eclipse.tracecompass.tmf.core.signal.TmfSignalManager.dispatchSignal(closeSignal);
    critPathMod.dispose();
}


function calculateVariance(array) {
    if (array.length === 0) return 0;
    var mean = array.reduce((a, b) => a + b) / array.length;
    var variance = array.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / array.length;
    return variance;																						 						  
}

// -------------------------
// Convert a status to a numeric value
// -------------------------
function getChar(status) {
				   
    if(status == "BLOCK_DEVICE"){
        return 0;
    } else if(status == "BLOCKED"){
        return 1;
    } else if(status == "DEFAULT"){
        return 2;
    } else if(status == "EPS"){
        return 3;
    } else if(status == "INTERRUPTED"){
        return 4;
    } else if(status == "IPI"){
        return 5;
    } else if(status =="NETWORK"){
        return 6;
    } else if(status == "PREEMPTED"){
        return 7;
    } else if(status == "RUNNING"){
        return 8;
    } else if(status == "TIMER"){
        return 9;
    } else if(status == "UNKNOWN"){
        return 11;
    } else if(status == "USER_INPUT"){
        return 12;
    } else {
        return 13;
    }
}

// -------------------------
// Utility to convert a vector to CSV
// -------------------------
function vectorToString(vector){
    var output = "";
    for(var f = 0; f < vector.length - 1; f++){
        output += vector[f] + ",";
    }
    output += vector[vector.length - 1];
    //print(output);
    return output;
}

