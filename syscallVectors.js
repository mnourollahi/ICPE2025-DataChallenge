// -------------------------
// Load necessary modules
// -------------------------
loadModule("/TraceCompass/Trace");
loadModule('/TraceCompass/Analysis');
loadModule('/System/Resources');
loadModule('/TraceCompass/View');
loadModule('/TraceCompass/TraceUI');

// -------------------------
// Paths and filenames
// -------------------------
const syscallPath = "/home/masoum/output/syscalls";
const vectorPath= "/home/masoum/output/vectors";

var followName = "elasticsearch";
var numTraces = 1;

print("start");
var d = new Date();
var t = d.getTime();

var syscallOut = createFile(syscallPath + t + ".txt");
var syscallHandle = writeLine(syscallOut, "ThreadID,WindowStartTime(s),Syscall Output");
var freqVectorOut = createFile(vectorPath +"Freq" + t +".csv");
var durVectorOut = createFile(vectorPath +"Dur" + t+ ".csv");
var statusVectorOut = createFile(vectorPath +"Status" + t+ ".csv");
var statusDurVectorOut = createFile(vectorPath +"StatusDur" + t+ ".csv");

// -------------------------
// Vector CSV headers
// -------------------------
header = "ThreadID,WindowStartTime(s),";
for(i=0;i<314;i++){
	header+=i+",";
}
header+=313;
var freqVectorHandle = writeLine(freqVectorOut, header);
var durVectorHandle = writeLine(durVectorOut, header);

// -------------------------
// Main loop over traces
// -------------------------
for(traceNum = 1; traceNum <= numTraces; traceNum++) { 
    print("Analyzing trace " + traceNum);

    var trace = getActiveTrace();
    if (trace == null) {
        print("Trace is null");
    } else {
        getData(trace);
    }
}

// -------------------------
// getData: Processes the trace with 100-second windowing
// -------------------------
function getData(trace) {
    var analysis = getTraceAnalysis(trace, 'OS Execution Graph');
    analysis.schedule();
    analysis.waitForCompletion();

    var osGraph = analysis.getGraph();
    var workers = osGraph.getWorkers();
    var iter = workers.iterator();
    print("Number of workers: " + workers.size());


    var batch = [];
    var batchSize = 1;  // Reduced batch size for better memory handling
    var batchIndex = 0;

    while (iter.hasNext()) {
        batch.push(iter.next());

        if (batch.length >= batchSize || !iter.hasNext()) {
            print("Processing batch " + batchIndex + " of size " + batch.length);
            try {
                processBatch(batch, trace);
            } catch (e) {
                print("Error processing batch " + batchIndex + ": " + e.message);
            }
            batch = null;  // Clear batch to free memory
            java.lang.System.gc();  // Trigger garbage collection
            batch = [];
            batchIndex++;
        }
    }
}

// -------------------------
// Process Batch of Syscall Events with Time Windowing
// -------------------------
function processBatch(batch, trace) {
    var windowSize = 100 * 1e9;

    batch.forEach(function(worker) {
        let name = worker.getName();
        if (name.contains("elasticsearch")) {
            let info = worker.getWorkerInformation();
            var threadid  = info.get('TID');
            processSyscalls(worker, trace, threadid, windowSize);
        }
        
    });
}

// -------------------------
// Process Syscall Events with Time Windowing
// -------------------------
function processSyscalls(worker, trace, threadid, windowSize) {
    var syscallEvents = [];
    var events = getEventIterator(trace);
    var eventCounter = 0;
    while (events.hasNext()) {
        var event = events.next();
        if (event.getName().startsWith("syscall")) {
            syscallEvents.push(event);
            eventCounter++;
            
            // Write in increments of 5000 to free memory
            if (eventCounter % 5000 === 0) {
                flushSyscalls(syscallEvents, threadid, windowSize);
                syscallEvents = [];
                java.lang.System.gc();  // Force garbage collection
            }
        }
    }
    
    // Flush any remaining events
    if (syscallEvents.length > 0) {
        flushSyscalls(syscallEvents, threadid, windowSize);
        syscallEvents = null;
        java.lang.System.gc();
    }
    // Flush the last window after all events


}


// -------------------------
// Process Syscall Events with Time Windowing
// -------------------------
function flushSyscalls(events, threadid, windowSize) {
    events.sort((a, b) => a.getTimestamp().toNanos() - b.getTimestamp().toNanos());
    var windowStart = events[0].getTimestamp().toNanos();
    var windowEnd = windowStart + windowSize;

    var freqVector = []
    var durVector = []
    for (var i = 0; i < 314; i++) {
       freqVector[i] = 0;
       durVector[i] = 0;
    }
    var lastSyscallTime = {};

    events.forEach(event => {
        var eventTime = event.getTimestamp().toNanos();
        if (eventTime > windowEnd) {
            writeVectorData(threadid, windowStart, freqVector, durVector);
            for (var i = 0; i < 314; i++) {
       	freqVector[i] = 0;
       	durVector[i] = 0;
   	    }
            windowStart = eventTime;
            windowEnd = windowStart + windowSize;
        }

        var syscallName = event.getName().substr(8);
        var syscallIndex = assignVal(syscallName);

        freqVector[syscallIndex]++;

        if (lastSyscallTime[syscallIndex] !== undefined) {
            durVector[syscallIndex] += (eventTime - lastSyscallTime[syscallIndex]);
        }
        lastSyscallTime[syscallIndex] = eventTime;

        var fields = event.getContent().getFields().iterator();
        while (fields.hasNext()) {
            var field = fields.next();
            if (field.getName() === "context._tid") {
                var eventTid = field.toString().substr(13);
                if (eventTid == threadid) {
                    writeLine(
                        syscallHandle,
                        threadid + "," + (windowStart / 1e9) + ',' + 'Time: ' + eventTime + ' CPU id: ' + event.getCPU() + ' Syscall: ' + event.getName().substr(8) + ' ' + event.getContent()
                    );
                        // Write frequency vector with a new line
    //writeLine(freqVectorHandle, threadid + "," + (windowStart / 1e9) + "," + vectorToString(freqVector));

    // Write duration vector with a new line
    //writeLine(durVectorHandle, threadid + "," + (windowStart / 1e9) + "," + vectorToString(durVector));
                }
            }
        }
    });

    writeVectorData(threadid, windowStart, freqVector, durVector);
}

// -------------------------
// Write Frequency and Duration Vectors
// -------------------------

function writeVectorData(threadid, windowStart, freqVector, durVector) {
    // Write frequency vector with a new line
    writeLine(freqVectorHandle, threadid + "," + (windowStart / 1e9) + "," + vectorToString(freqVector));

    // Write duration vector with a new line
    writeLine(durVectorHandle, threadid + "," + (windowStart / 1e9) + "," + vectorToString(durVector));
}




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


/**
 * Assign a syscall type to an index.
 */
function assignVal(syscall) {
    const syscallMap = {
         sys_read: 0,
	 write: 1,
        open: 2,
        close: 3,
        stat: 4,
        fstat: 5,
        lstat: 6,
        poll: 7,
        lseek: 8,
        mmap: 9,
        mprotect: 10,
        munmap: 11,
        brk: 12,
        rt_sigaction: 13,
        rt_sigprocmask: 14,
        rt_sigreturn: 15,
        ioctl: 16,
        pread64: 17,
        pwrite64: 18,
        readv: 19,
        writev: 20,
        access: 21,
        pipe: 22,
        select: 23,
        sched_yield: 24,
        mremap: 25,
        msync: 26,
        mincore: 27,
        madvise: 28,
        shmget: 29,
        shmat: 30,
        shmctl: 31,
        dup: 32,
        dup2: 33,
        pause: 34,
        nanosleep: 35,
        getitimer: 36,
        alarm: 37,
        setitimer: 38,
        getpid: 39,
        sendfile: 40,
        socket: 41,
        connect: 42,
        accept: 43,
        sendto: 44,
        recvfrom: 45,
        sendmsg: 46,
        recvmsg: 47,
        shutdown: 48,
        bind: 49,
        listen: 50,
        getsockname: 51,
        getpeername: 52,
        socketpair: 53,
        setsockopt: 54,
        getsockopt: 55,
        clone: 56,
        fork: 57,
        vfork: 58,
        execve: 59,
        exit: 60,
        wait4: 61,
        kill: 62,
        uname: 63,
        semget: 64,
        semop: 65,
        semctl: 66,
        shmdt: 67,
        msgget: 68,
        msgsnd: 69,
        msgrcv: 70,
        msgctl: 71,
        fcntl: 72,
        flock: 73,
        fsync: 74,
        fdatasync: 75,
        truncate: 76,
        ftruncate: 77,
        getdents: 78,
        getcwd: 79,
        chdir: 80,
        fchdir: 81,
        rename: 82,
        mkdir: 83,
        rmdir: 84,
        creat: 85,
        link: 86,
        unlink: 87,
        symlink: 88,
        readlink: 89,
        chmod: 90,
        fchmod: 91,
        chown: 92,
        fchown: 93,
        lchown: 94,
        umask: 95,
        gettimeofday: 96,
        getrlimit: 97,
        getrusage: 98,
        sysinfo: 99,
        times: 100,
        ptrace: 101,
        getuid: 102,
        syslog: 103,
        getgid: 104,
        setuid: 105,
        setgid: 106,
        geteuid: 107,
        getegid: 108,
        setpgid: 109,
        getppid: 110,
        getpgrp: 111,
        setsid: 112,
        setreuid: 113,
        setregid: 114,
        getgroups: 115,
        setgroups: 116,
        setresuid: 117,
        getresuid: 118,
        setresgid: 119,
        getresgid: 120,
        getpgid: 121,
        setfsuid: 122,
        setfsgid: 123,
        getsid: 124,
        capget: 125,
        capset: 126,
        rt_sigpending: 127,
        rt_sigtimedwait: 128,
        rt_sigqueueinfo: 129,
        rt_sigsuspend: 130,
        sigaltstack: 131,
        utime: 132,
        mknod: 133,
        uselib: 134,
        personality: 135,
        ustat: 136,
        statfs: 137,
        fstatfs: 138,
        sysfs: 139,
        getpriority: 140,
        setpriority: 141,
        sched_setparam: 142,
        sched_getparam: 143,
        sched_setscheduler: 144,
        sched_getscheduler: 145,
        sched_get_priority_max: 146,
        sched_get_priority_min: 147,
        sched_rr_get_interval: 148,
        mlock: 149,
        munlock: 150,
        mlockall: 151,
        munlockall: 152,
        vhangup: 153,
        modify_ldt: 154,
        pivot_root: 155,
        _sysctl: 156,
        prctl: 157,
        arch_prctl: 158,
        adjtimex: 159,
        setrlimit: 160,
        chroot: 161,
        sync: 162,
        acct: 163,
        settimeofday: 164,
        mount: 165,
        umount2: 166,
        swapon: 167,
        swapoff: 168,
        reboot: 169,
        sethostname: 170,
        setdomainname: 171,
        iopl: 172,
        ioperm: 173,
        create_module: 174,
        init_module: 175,
        delete_module: 176,
        get_kernel_syms: 177,
        query_module: 178,
        quotactl: 179,
        nfsservctl: 180,
        getpmsg: 181,
        putpmsg: 182,
        afs_syscall: 183,
        tuxcall: 184,
        security: 185,
        gettid: 186,
        readahead: 187,
        setxattr: 188,
        lsetxattr: 189,
        fsetxattr: 190,
        getxattr: 191,
        lgetxattr: 192,
        fgetxattr: 193,
        listxattr: 194,
        llistxattr: 195,
        flistxattr: 196,
        removexattr: 197,
        lremovexattr: 198,
        fremovexattr: 199,
        tkill: 200,
        time: 201,
        futex: 202,
        sched_setaffinity: 203,
        sched_getaffinity: 204,
        set_thread_area: 205,
        io_setup: 206,
        io_destroy: 207,
        io_getevents: 208,
        io_submit: 209,
        io_cancel: 210,
        get_thread_area: 211,
        lookup_dcookie: 212,
        epoll_create: 213,
        epoll_ctl_old: 214,
        epoll_wait_old: 215,
        remap_file_pages: 216,
        getdents64: 217,
        set_tid_address: 218,
        restart_syscall: 219,
        semtimedop: 220,
        fadvise64: 221,
        timer_create: 222,
        timer_settime: 223,
        timer_gettime: 224,
        timer_getoverrun: 225,
        timer_delete: 226,
        clock_settime: 227,
        clock_gettime: 228,
        clock_getres: 229,
        clock_nanosleep: 230,
        exit_group: 231,
        epoll_wait: 232,
        epoll_ctl: 233,
        tgkill: 234,
        utimes: 235,
        vserver: 236,
        mbind: 237,
        set_mempolicy: 238,
        get_mempolicy: 239,
        mq_open: 240,
        mq_unlink: 241,
        mq_timedsend: 242,
        mq_timedreceive: 243,
        mq_notify: 244,
        mq_getsetattr: 245,
        kexec_load: 246,
        waitid: 247,
        add_key: 248,
        request_key: 249,
        keyctl: 250,
        ioprio_set: 251,
        ioprio_get: 252,
        inotify_init: 253,
        inotify_add_watch: 254,
        inotify_rm_watch: 255,
        migrate_pages: 256,
        openat: 257,
        mkdirat: 258,
        mknodat: 259,
        fchownat: 260,
        futimesat: 261,
        newfstatat: 262,
        unlinkat: 263,
        renameat: 264,
        linkat: 265,
        symlinkat: 266,
        readlinkat: 267,
        fchmodat: 268,
        faccessat: 269,
        pselect6: 270,
        ppoll: 271,
        unshare: 272,
        set_robust_list: 273,
        get_robust_list: 274,
        splice: 275,
        tee: 276,
        sync_file_range: 277,
        vmsplice: 278,
        move_pages: 279,
        utimensat: 280,
        epoll_pwait: 281,
        signalfd: 282,
        timerfd_create: 283,
        eventfd: 284,
        fallocate: 285,
        timerfd_settime: 286,
        timerfd_gettime: 287,
        accept4: 288,
        signalfd4: 289,
        eventfd2: 290,
        epoll_create1: 291,
        dup3: 292,
        pipe2: 293,
        inotify_init1: 294,
        preadv: 295,
        pwritev: 296,
        rt_tgsigqueueinfo: 297,
        perf_event_open: 298,
        recvmmsg: 299,
        fanotify_init: 300,
        fanotify_mark: 301,
        prlimit64: 302,
        name_to_handle_at: 303,
        open_by_handle_at: 304,
        clock_adjtime: 305,
        syncfs: 306,
        sendmmsg: 307,
        setns: 308,
        getcpu: 309,
        process_vm_readv: 310,
        process_vm_writev: 311,
        kcmp: 312,
        finit_module: 313,
        default: 314 // Use for unknown syscalls
    };
    return syscallMap[syscall] || syscallMap.default;
}

