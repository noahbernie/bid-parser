"""init_full_schema

Revision ID: 27d719695777
Revises:
Create Date: 2026-05-01 08:20:20.027286

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '27d719695777'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.execute("""
        CREATE TABLE organizations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            photo_url TEXT,
            microsoft_tenant_id TEXT,
            subscription_status TEXT NOT NULL DEFAULT 'active',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID REFERENCES organizations(id),
            email TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            avatar_url TEXT,
            password_hash TEXT,
            microsoft_oid TEXT,
            role TEXT NOT NULL DEFAULT 'member',
            onboarded_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE projects (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id),
            name TEXT NOT NULL,
            location TEXT,
            status TEXT NOT NULL DEFAULT 'in_progress',
            created_by_user_id UUID NOT NULL REFERENCES users(id),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE user_projects (
            user_id UUID NOT NULL REFERENCES users(id),
            project_id UUID NOT NULL REFERENCES projects(id),
            access_level TEXT NOT NULL DEFAULT 'viewer',
            assigned_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            assigned_by_user_id UUID REFERENCES users(id),
            PRIMARY KEY (user_id, project_id)
        )
    """)

    op.execute("""
        CREATE TABLE organization_invites (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id),
            invited_by_user_id UUID NOT NULL REFERENCES users(id),
            email TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'member',
            status TEXT NOT NULL DEFAULT 'pending',
            token TEXT NOT NULL UNIQUE,
            grant_all_projects BOOLEAN NOT NULL DEFAULT false,
            expires_at TIMESTAMPTZ NOT NULL,
            accepted_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE subscriptions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id),
            plan TEXT NOT NULL DEFAULT 'monthly',
            monthly_fee_cents INTEGER NOT NULL DEFAULT 300000,
            billing_start_date DATE NOT NULL,
            next_renewal_date DATE NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE audit_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id),
            user_id UUID NOT NULL REFERENCES users(id),
            action TEXT NOT NULL,
            entity_type TEXT,
            entity_id UUID,
            metadata JSONB,
            ip_address TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE usage_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id),
            month DATE NOT NULL,
            jobs_uploaded INTEGER NOT NULL DEFAULT 0,
            streets_parsed INTEGER NOT NULL DEFAULT 0,
            takeoffs_generated INTEGER NOT NULL DEFAULT 0,
            takeoffs_approved INTEGER NOT NULL DEFAULT 0,
            exports_generated INTEGER NOT NULL DEFAULT 0,
            active_users INTEGER NOT NULL DEFAULT 0,
            recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE jobs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            project_id UUID NOT NULL REFERENCES projects(id),
            organization_id UUID NOT NULL REFERENCES organizations(id),
            uploaded_by_user_id UUID NOT NULL REFERENCES users(id),
            job_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'uploaded',
            parse_error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE job_media (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id UUID NOT NULL REFERENCES jobs(id),
            s3_bucket TEXT NOT NULL,
            s3_key TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size_bytes BIGINT NOT NULL,
            uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE bid_parse_results (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id UUID NOT NULL UNIQUE REFERENCES jobs(id),
            bid_number TEXT,
            project_name TEXT,
            city TEXT,
            state TEXT,
            work_types TEXT[],
            estimated_cost NUMERIC,
            bid_due_date DATE,
            total_pages INTEGER,
            selected_pages INTEGER,
            selected_page_numbers INTEGER[],
            total_streets INTEGER,
            chunks_processed INTEGER,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE parser_stage_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id UUID NOT NULL REFERENCES jobs(id),
            stage TEXT NOT NULL,
            stage_order INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'success',
            street_count_in INTEGER,
            street_count_out INTEGER,
            streets_dropped INTEGER,
            pages_processed INTEGER,
            pages_selected INTEGER,
            selected_page_numbers INTEGER[],
            duration_ms INTEGER,
            error_message TEXT,
            raw_log_s3_key TEXT,
            metadata JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE streets_raw (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id UUID NOT NULL REFERENCES jobs(id),
            parser_stage_log_id UUID REFERENCES parser_stage_logs(id),
            main_street TEXT NOT NULL,
            from_street TEXT,
            to_street TEXT,
            work_type TEXT,
            page INTEGER,
            source TEXT NOT NULL DEFAULT 'gemini-pro',
            tags TEXT[],
            location TEXT,
            confidence TEXT NOT NULL DEFAULT 'high',
            low_confidence_reason TEXT,
            is_active BOOLEAN NOT NULL DEFAULT true,
            validated BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE streets_ledger (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            street_raw_id UUID NOT NULL REFERENCES streets_raw(id),
            user_id UUID NOT NULL REFERENCES users(id),
            field TEXT NOT NULL,
            old_value TEXT,
            new_value TEXT,
            change_type TEXT NOT NULL,
            changed_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE streets_final (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id UUID NOT NULL REFERENCES jobs(id),
            street_raw_id UUID REFERENCES streets_raw(id),
            main_street TEXT NOT NULL,
            from_street TEXT,
            to_street TEXT,
            work_type TEXT,
            tags TEXT[],
            source TEXT NOT NULL DEFAULT 'gemini-pro',
            validated_by_user_id UUID REFERENCES users(id),
            validated_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE takeoffs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            project_id UUID NOT NULL REFERENCES projects(id),
            material_type TEXT NOT NULL,
            total_area_sqft NUMERIC,
            total_area_sqm NUMERIC,
            total_length_ft NUMERIC,
            status TEXT NOT NULL DEFAULT 'draft',
            approved_by_user_id UUID REFERENCES users(id),
            approved_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE street_takeoffs_raw (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            work_types TEXT[] NOT NULL,
            model_version TEXT NOT NULL,
            centerline JSONB,
            total_length_m NUMERIC,
            mpp DOUBLE PRECISION,
            polygon JSONB,
            raw_polygons JSONB,
            experimental_polygons JSONB,
            capped_polygons JSONB,
            area_sqft NUMERIC,
            area_sqm NUMERIC,
            length_ft NUMERIC,
            length_m NUMERIC,
            avg_width_ft NUMERIC,
            post_process_params JSONB,
            mask_cache_s3_bucket TEXT,
            mask_cache_s3_key TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE street_takeoff_raw_streets (
            street_takeoff_raw_id UUID NOT NULL REFERENCES street_takeoffs_raw(id),
            street_final_id UUID NOT NULL REFERENCES streets_final(id),
            material_type TEXT NOT NULL,
            PRIMARY KEY (street_takeoff_raw_id, street_final_id)
        )
    """)

    op.execute("""
        CREATE TABLE tiles (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            street_takeoff_raw_id UUID NOT NULL REFERENCES street_takeoffs_raw(id),
            tile_idx INTEGER NOT NULL,
            center_lat DOUBLE PRECISION NOT NULL,
            center_lng DOUBLE PRECISION NOT NULL,
            mpp DOUBLE PRECISION NOT NULL,
            width_px INTEGER NOT NULL DEFAULT 1280,
            height_px INTEGER NOT NULL DEFAULT 1280,
            zoom INTEGER NOT NULL DEFAULT 20,
            scale INTEGER NOT NULL DEFAULT 2,
            point_range_start INTEGER NOT NULL,
            point_range_end INTEGER NOT NULL,
            s3_bucket TEXT,
            s3_key TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE street_takeoffs_ledger (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            street_takeoff_raw_id UUID NOT NULL REFERENCES street_takeoffs_raw(id),
            street_takeoff_staging_id UUID,
            user_id UUID NOT NULL REFERENCES users(id),
            change_type TEXT NOT NULL,
            field TEXT,
            old_value JSONB,
            new_value JSONB,
            post_process_params JSONB,
            changed_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE street_takeoffs_staging (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            takeoff_id UUID NOT NULL REFERENCES takeoffs(id),
            street_takeoff_raw_id UUID NOT NULL REFERENCES street_takeoffs_raw(id),
            street_final_id UUID NOT NULL REFERENCES streets_final(id),
            material_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            polygon_variant TEXT NOT NULL DEFAULT 'polygon',
            polygon JSONB,
            area_sqft NUMERIC NOT NULL DEFAULT 0,
            area_sqm NUMERIC,
            length_ft NUMERIC,
            length_m NUMERIC,
            avg_width_ft NUMERIC,
            post_process_params JSONB,
            notes TEXT,
            approved_by_user_id UUID REFERENCES users(id),
            approved_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE street_takeoffs_final (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            street_takeoff_staging_id UUID NOT NULL REFERENCES street_takeoffs_staging(id),
            takeoff_id UUID NOT NULL REFERENCES takeoffs(id),
            street_final_id UUID NOT NULL REFERENCES streets_final(id),
            material_type TEXT NOT NULL,
            polygon_variant TEXT NOT NULL,
            polygon JSONB,
            area_sqft NUMERIC NOT NULL,
            area_sqm NUMERIC,
            length_ft NUMERIC,
            length_m NUMERIC,
            avg_width_ft NUMERIC,
            notes TEXT,
            approved_by_user_id UUID NOT NULL REFERENCES users(id),
            snapshot_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE model_versions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            version_tag TEXT NOT NULL UNIQUE,
            encoder TEXT NOT NULL,
            in_channels INTEGER NOT NULL DEFAULT 4,
            num_classes INTEGER NOT NULL DEFAULT 7,
            s3_weights_key TEXT NOT NULL,
            trained_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            val_mean_iou NUMERIC,
            notes TEXT,
            is_active BOOLEAN NOT NULL DEFAULT false
        )
    """)

    op.execute("""
        CREATE TABLE model_inference_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            street_takeoff_raw_id UUID NOT NULL REFERENCES street_takeoffs_raw(id),
            model_version_id UUID NOT NULL REFERENCES model_versions(id),
            num_tiles INTEGER,
            inference_duration_ms INTEGER,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE exports (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            project_id UUID NOT NULL REFERENCES projects(id),
            created_by_user_id UUID NOT NULL REFERENCES users(id),
            format TEXT NOT NULL,
            scope TEXT NOT NULL DEFAULT 'full_project',
            takeoff_ids UUID[],
            s3_bucket TEXT NOT NULL,
            s3_key TEXT NOT NULL,
            file_size_bytes BIGINT,
            status TEXT NOT NULL DEFAULT 'generating',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            completed_at TIMESTAMPTZ
        )
    """)

    # Indexes
    op.execute("CREATE INDEX idx_jobs_project_id ON jobs(project_id)")
    op.execute("CREATE INDEX idx_jobs_organization_id ON jobs(organization_id)")
    op.execute("CREATE INDEX idx_jobs_status ON jobs(status)")
    op.execute("CREATE INDEX idx_parser_stage_logs_job_id ON parser_stage_logs(job_id)")
    op.execute("CREATE INDEX idx_parser_stage_logs_stage ON parser_stage_logs(stage)")
    op.execute("CREATE INDEX idx_streets_raw_job_id ON streets_raw(job_id)")
    op.execute("CREATE INDEX idx_streets_raw_confidence ON streets_raw(confidence)")
    op.execute("CREATE INDEX idx_streets_raw_is_active ON streets_raw(is_active)")
    op.execute("CREATE INDEX idx_streets_raw_validated ON streets_raw(validated)")
    op.execute("CREATE INDEX idx_streets_final_job_id ON streets_final(job_id)")
    op.execute("CREATE INDEX idx_street_takeoffs_staging_status ON street_takeoffs_staging(status)")
    op.execute("CREATE INDEX idx_audit_log_organization_id ON audit_log(organization_id)")
    op.execute("CREATE INDEX idx_audit_log_created_at ON audit_log(created_at DESC)")


def downgrade() -> None:
    op.execute("""
        DROP TABLE IF EXISTS exports, model_inference_log, model_versions,
        street_takeoffs_final, street_takeoffs_staging, street_takeoffs_ledger,
        tiles, street_takeoff_raw_streets, street_takeoffs_raw, takeoffs,
        streets_final, streets_ledger, streets_raw, parser_stage_logs,
        bid_parse_results, job_media, jobs, usage_log, audit_log,
        subscriptions, organization_invites, user_projects, projects,
        users, organizations CASCADE
    """)
