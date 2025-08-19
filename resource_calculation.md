# Resource Calculation for Processing 69,996 Samples

## Current Performance Data
- **Average time per sample**: 551 seconds (9.2 minutes)
- **Memory per sample**: 2.7GB RAM, 3.6GB GPU memory
- **Success rate**: 60% (based on recent run)

## Time Requirements

### Sequential Processing (Not Recommended)
- Total time: 69,996 × 551 seconds = 38,567,796 seconds
- = 10,713 hours = 446 days = 1.2 years ❌

### Parallel Processing (Recommended)
- **Chunk size**: 50 samples (reduced from 100 due to memory constraints)
- **Number of chunks**: 69,996 ÷ 50 = 1,400 chunks
- **Time per chunk**: 50 × 551 seconds = 27,550 seconds = 7.7 hours
- **With 10 parallel chunks**: 27,550 ÷ 10 = 2,755 hours = 115 days = 3.8 months

## Memory Requirements

### Per Chunk (50 samples)
- **RAM**: 50 × 2.7GB = 135GB
- **GPU Memory**: 50 × 3.6GB = 180GB

### Total Resources Needed
- **Total RAM**: 135GB × 10 parallel chunks = 1,350GB = 1.35TB
- **Total GPU Memory**: 180GB × 10 parallel chunks = 1,800GB = 1.8TB

## Recommended Configuration

### Option 1: Conservative (5 parallel chunks)
- **Time**: 27,550 ÷ 5 = 5,510 hours = 230 days = 7.6 months
- **RAM**: 135GB × 5 = 675GB
- **GPU Memory**: 180GB × 5 = 900GB

### Option 2: Aggressive (20 parallel chunks)
- **Time**: 27,550 ÷ 20 = 1,378 hours = 57 days = 1.9 months
- **RAM**: 135GB × 20 = 2,700GB = 2.7TB
- **GPU Memory**: 180GB × 20 = 3,600GB = 3.6TB

## Cluster Requirements

### Minimum Viable Setup
- **Nodes**: 5 nodes with 4× A100 80GB each
- **Total GPUs**: 20 × A100 80GB = 1,600GB GPU memory
- **Total RAM**: 5 × 256GB = 1,280GB RAM
- **Processing time**: ~2 months

### Optimal Setup
- **Nodes**: 10 nodes with 4× A100 80GB each
- **Total GPUs**: 40 × A100 80GB = 3,200GB GPU memory
- **Total RAM**: 10 × 256GB = 2,560GB RAM
- **Processing time**: ~1 month

## Cost Estimation

### Hardware Costs (if purchasing)
- **A100 80GB**: ~$10,000 each
- **Server with 4× A100**: ~$50,000 each
- **10 servers**: ~$500,000

### Cloud Costs (if using cloud)
- **AWS p4d.24xlarge**: ~$32/hour per instance
- **10 instances for 1 month**: ~$230,000

## Recommendations

1. **Start with 5 parallel chunks** to test system stability
2. **Monitor memory usage** and adjust chunk size if needed
3. **Use checkpointing** to resume from failures
4. **Implement result aggregation** to combine chunk outputs
5. **Consider using smaller models** (1.5B instead of 3B) for faster processing

## Chunk Configuration

### Recommended Chunk Settings
```bash
# Chunk size: 50 samples
# Number of chunks: 1,400
# Parallel chunks: 5-20 (depending on resources)

# Example submission
./submit_chunks_parallel.sh 0 4 5  # Submit first 5 chunks with max 5 parallel
```

### Memory-Optimized Settings
- **Chunk size**: 50 samples (reduced from 100)
- **MCTS iterations**: 100 (increased from 50)
- **Timeout**: 10 minutes per sample
- **Batch size**: 1 (already optimized)


