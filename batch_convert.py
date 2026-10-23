#!/usr/bin/env python3
"""
batch_convert_vulstream.py

Script 3 — standalone batch job. Converts the REAL production archive:

    s3://ltds-storage-gateway-tvnews/vulstream/mpeg4/{YYYY}/{MM}/{CHANNEL}/{YYYYMMDD}{CHANNEL}.mp4

into whole-broadcast HLS, written into a TESTING prefix in the SAME bucket:

    s3://ltds-storage-gateway-tvnews/vulstream/hls/zzz-RS-Testing/{YYYY}/{MM}/{CHANNEL}/{YYYYMMDD}{CHANNEL}-{broadcast_id}/
        {YYYYMMDD}{CHANNEL}_640p.m3u8
        {YYYYMMDD}{CHANNEL}_640p_000.ts
        {YYYYMMDD}{CHANNEL}_640p_001.ts
        ...

Mirrors the source's own Year -> Month -> Channel folder structure exactly,
just moved from mpeg4/ to hls/zzz-RS-Testing/, with broadcast_id (nanoid)
appended to the final folder name for DB traceability / S3 search.

This does NOT touch mpeg4/ (read-only, source archive). It only ever writes
under hls/zzz-RS-Testing/.

This script calls the existing, UNMODIFIED s3_to_hls.py as a subprocess,
always with --no-clip (whole broadcast, not the 00:00:40-00:02:50 default).

Usage — single broadcast (recommended first run):
    python batch_convert_vulstream_to_hls.py \
        --bucket ltds-storage-gateway-tvnews \
        --only-key vulstream/mpeg4/2007/10/ABC/20071008ABC.mp4 \
        --dry-run

Usage — one whole year (safer than the entire archive at once):
    python batch_convert_vulstream_to_hls.py \
        --bucket ltds-storage-gateway-tvnews \
        --only-year 2007

Usage — everything, 1968 through 2026 (only after the above are verified):
    python batch_convert_vulstream_to_hls.py \
        --bucket ltds-storage-gateway-tvnews

Requires:
    pip install boto3

DB access is via RDS Data API (boto3 rds-data client) -- this cluster has
no public MySQL endpoint, so a direct pymysql/mysql-connector connection
will NOT work here. Credentials come from the resource/secret ARNs baked
in below (same ones used for the manual CLI DESCRIBE/SHOW TABLES checks),
not from env vars. The IAM role/user running this script needs
rds-data:ExecuteStatement and secretsmanager:GetSecretValue on that secret.
"""

import argparse
import logging
import os
import re
import subprocess
import sys

import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_REGION = "us-east-1"

SRC_PREFIX = "vulstream/mpeg4"
DST_PREFIX_ROOT = "vulstream/hls/zzz-RS-Testing"

# vulstream/mpeg4/{YYYY}/{MM}/{CHANNEL}/{YYYYMMDD}{CHANNEL}.mp4
SRC_KEY_RE = re.compile(
    rf"^{re.escape(SRC_PREFIX)}/"
    r"(?P<year>\d{4})/(?P<month>\d{2})/(?P<channel>[A-Za-z0-9]+)/"
    r"(?P<yyyymmdd>\d{8})(?P<channel2>[A-Za-z0-9]+)\.mp4$"
)

# Confirmed 2026-08-25 via CloudShell + RDS Data API against prod_tvnews_aws.
# No public MySQL endpoint on this cluster -- Data API (HTTPS) is the only
# route in, same as the manual DESCRIBE/SHOW TABLES commands used to confirm
# the schema below.
DB_RESOURCE_ARN = "arn:aws:rds:us-east-1:189784013121:cluster:prod-tvnews-aws-aurora"
DB_SECRET_ARN = "arn:aws:secretsmanager:us-east-1:189784013121:secret:rds!cluster-1ca10d10-b44d-424e-bce6-23b43f714a57-iauukQ"
DB_NAME = "prod_tvnews_aws"


def get_rds_data_client():
    return boto3.client("rds-data", region_name=DEFAULT_REGION)


def lookup_broadcast_id(rds_data, yyyymmdd: str, channel: str) -> str | None:
    """
    Resolve broadcast_id (nanoid) for a given date + channel/network, via
    RDS Data API (execute_statement) rather than a direct MySQL connection.

    Confirmed against real prod_tvnews_aws schema (2026-08-25):
      broadcasts.broadcast_id     char(10), PK
      broadcasts.broadcast_date   date
      broadcasts.network          tinyint  -> FK into net_dist.net_dist_id
      net_dist.net_dist_id        tinyint, PK, auto_increment
      net_dist.net_dist_value     varchar(100)  (e.g. "ABC") -- same table
                                   is also used for broadcasts.distributor,
                                   so net_dist holds both networks and
                                   distributors together. Values confirmed
                                   clean uppercase abbreviations matching
                                   S3 folder names (ABC, CBS, NBC, ...),
                                   except a few multi-word ones (e.g.
                                   "News Nation", "BBC America") -- these
                                   won't match the channel-folder regex and
                                   will be skipped/logged, not silently
                                   mismatched.
    """
    date_fmt = f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
    sql = """
        SELECT b.broadcast_id
        FROM broadcasts b
        JOIN net_dist n ON b.network = n.net_dist_id
        WHERE b.broadcast_date = :broadcast_date
          AND n.net_dist_value = :channel
        LIMIT 1
    """
    resp = rds_data.execute_statement(
        resourceArn=DB_RESOURCE_ARN,
        secretArn=DB_SECRET_ARN,
        database=DB_NAME,
        sql=sql,
        parameters=[
            {"name": "broadcast_date", "value": {"stringValue": date_fmt}},
            {"name": "channel", "value": {"stringValue": channel}},
        ],
    )
    records = resp.get("records", [])
    if not records:
        return None
    return records[0][0].get("stringValue")


def list_source_mp4s(s3, bucket: str, only_key: str | None, only_year: str | None):
    """Yield source .mp4 keys under vulstream/mpeg4/, optionally scoped to one year."""
    if only_key:
        yield only_key
        return

    prefix = f"{SRC_PREFIX}/{only_year}/" if only_year else f"{SRC_PREFIX}/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".mp4"):
                yield key


def build_dst_prefix(src_key: str, broadcast_id: str | None) -> str | None:
    m = SRC_KEY_RE.match(src_key)
    if not m:
        log.warning("Skipping unrecognized key format: %s", src_key)
        return None
    year = m.group("year")
    month = m.group("month")
    channel = m.group("channel")
    yyyymmdd = m.group("yyyymmdd")

    folder_name = f"{yyyymmdd}{channel}"
    if broadcast_id:
        folder_name += f"-{broadcast_id}"

    return f"{DST_PREFIX_ROOT}/{year}/{month}/{channel}/{folder_name}"


def already_converted(s3, bucket: str, dst_prefix: str) -> bool:
    """Skip broadcasts that already have output (e.g. a .m3u8 already there) -- lets you
    safely re-run this script over the same year without redoing finished work."""
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"{dst_prefix}/", MaxKeys=1)
    return resp.get("KeyCount", 0) > 0


def convert_one(bucket: str, src_key: str, dst_prefix: str, s3_to_hls_path: str) -> None:
    cmd = [
        sys.executable, s3_to_hls_path,
        "--src-bucket", bucket,
        "--src-key", src_key,
        "--dst-bucket", bucket,          # same bucket, different prefix
        "--dst-prefix", dst_prefix,
        "--no-clip",
    ]
    log.info("Converting %s -> s3://%s/%s/", src_key, bucket, dst_prefix)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("Conversion FAILED for %s (exit %d)", src_key, result.returncode)
    else:
        log.info("Done: %s", src_key)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True, help="Same bucket for source and destination, e.g. ltds-storage-gateway-tvnews")
    p.add_argument("--only-key", default=None, help="Convert just this one source key (e.g. for a single-broadcast test)")
    p.add_argument("--only-year", default=None, help="Restrict to one year folder, e.g. 2007")
    p.add_argument("--s3-to-hls-path", default="./s3_to_hls.py")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip broadcasts that already have output in the destination prefix")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve broadcast_id + print the dst-prefix, don't actually convert")
    args = p.parse_args()

    session = boto3.Session(region_name=DEFAULT_REGION)
    s3 = session.client("s3")
    rds_data = get_rds_data_client()

    converted, skipped, failed = 0, 0, 0

    for src_key in list_source_mp4s(s3, args.bucket, args.only_key, args.only_year):
        m = SRC_KEY_RE.match(src_key)
        if not m:
            log.warning("Skipping unrecognized key format: %s", src_key)
            skipped += 1
            continue

        yyyymmdd = m.group("yyyymmdd")
        channel = m.group("channel")

        broadcast_id = lookup_broadcast_id(rds_data, yyyymmdd, channel)
        if not broadcast_id:
            log.warning("No broadcast_id found for %s (%s/%s) -- continuing WITHOUT id in the name",
                        src_key, yyyymmdd, channel)

        dst_prefix = build_dst_prefix(src_key, broadcast_id)
        if not dst_prefix:
            skipped += 1
            continue

        if args.skip_existing and already_converted(s3, args.bucket, dst_prefix):
            log.info("Already converted, skipping: %s", dst_prefix)
            skipped += 1
            continue

        if args.dry_run:
            log.info("[DRY RUN] %s -> s3://%s/%s/", src_key, args.bucket, dst_prefix)
            continue

        convert_one(args.bucket, src_key, dst_prefix, args.s3_to_hls_path)
        converted += 1

    log.info("Finished. converted=%d skipped=%d failed=%d", converted, skipped, failed)


if __name__ == "__main__":
    main()