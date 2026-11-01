-- Legacy ERP schema. Uppercase names are quoted so Postgres keeps their case.
-- No foreign keys are declared: relationships are by convention only.
-- Dates are integers in YYYYMMDD form. Code columns are space padded.

CREATE TABLE "GLPERD" (
    "PRD_ID"   integer NOT NULL,
    "STRT_DT"  integer NOT NULL,
    "END_DT"   integer NOT NULL,
    "CLOSE_DT" integer,
    "STAT_CD"  char(1) NOT NULL
);

CREATE TABLE "GLCOA" (
    "ACCT_CD"  char(8) NOT NULL,
    "ACCT_NM"  text NOT NULL,
    "ACCT_TYP" text NOT NULL,
    "ACTV_FLG" char(1) NOT NULL
);

CREATE TABLE "GLJRNH" (
    "JRNL_NO"     integer NOT NULL,
    "PRD_ID"      integer NOT NULL,
    "POST_DT"     integer NOT NULL,
    "POST_FLG"    char(1) NOT NULL,
    "REF_JRNL_NO" integer,
    "SRC_CD"      char(2) NOT NULL,
    "SRC_REF"     text,
    "MEMO"        text
);

CREATE TABLE "GLJRNL" (
    "JRNL_NO" integer NOT NULL,
    "LN_SEQ"  integer NOT NULL,
    "ACCT_CD" char(8) NOT NULL,
    "DR_AMT"  numeric(14,2) NOT NULL,
    "CR_AMT"  numeric(14,2) NOT NULL
);

CREATE TABLE "APVEND" (
    "VEND_CD"  char(8) NOT NULL,
    "VEND_NM"  text NOT NULL,
    "ACTV_FLG" char(1) NOT NULL
);

CREATE TABLE "APINVH" (
    "INV_ID"   integer NOT NULL,
    "VEND_CD"  char(8) NOT NULL,
    "INV_NO"   text NOT NULL,
    "INV_DT"   integer NOT NULL,
    "INV_AMT"  numeric(14,2) NOT NULL,
    "STAT_CD"  char(1) NOT NULL,
    "JRNL_NO"  integer NOT NULL
);

CREATE TABLE "APPAYM" (
    "PAY_ID"  integer NOT NULL,
    "INV_ID"  integer NOT NULL,
    "PAY_DT"  integer NOT NULL,
    "PAY_AMT" numeric(14,2) NOT NULL
);

CREATE TABLE "SYSDICT" (
    "TBL_NM" text NOT NULL,
    "COL_NM" text NOT NULL,
    "DESCR"  text NOT NULL
);
