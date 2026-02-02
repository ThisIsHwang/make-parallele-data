from synth_parallel.sampling.bucketer import BucketSampler, find_bucket


def test_bucket_sampler_quota_and_spillover():
    boundaries = [0, 5, 10]
    sampler = BucketSampler(boundaries, total_target=4, per_bucket_quota=2, spillover_capacity=10)

    for i in range(10):
        bucket_id = find_bucket(boundaries, i)
        sampler.add(bucket_id, {"source_id": str(i)})

    records = sampler.finalize()
    assert len(records) == 4
