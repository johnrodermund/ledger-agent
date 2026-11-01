# Data dictionary

Loaded into SYSDICT. One row per documented column.

| TBL_NM | COL_NM | DESCR |
|---|---|---|
| GLPERD | PRD_ID | Accounting period number |
| GLPERD | STRT_DT | First day of the period, YYYYMMDD |
| GLPERD | END_DT | Last day of the period, YYYYMMDD |
| GLPERD | STAT_CD | Period status: O open, C closed |
| GLCOA | ACCT_CD | Account code, space padded to 8 characters |
| GLCOA | ACCT_NM | Account name |
| GLCOA | ACCT_TYP | Account type: ASSET, LIABILITY, EXPENSE |
| GLCOA | ACTV_FLG | Active flag: Y active, N retired |
| GLJRNH | JRNL_NO | Journal number |
| GLJRNH | PRD_ID | Accounting period the journal belongs to |
| GLJRNH | POST_DT | Posting date, YYYYMMDD |
| GLJRNH | POST_FLG | Posting flag: P posted, U unposted |
| GLJRNH | SRC_CD | Source of the journal: AP payables, GL general ledger |
| GLJRNH | SRC_REF | Name of the user who keyed the entry |
| GLJRNH | MEMO | Free-text description |
| GLJRNL | JRNL_NO | Journal number of the parent header |
| GLJRNL | LN_SEQ | Line sequence within the journal |
| GLJRNL | ACCT_CD | Account the line posts to |
| GLJRNL | DR_AMT | Debit amount |
| GLJRNL | CR_AMT | Credit amount |
| APVEND | VEND_CD | Vendor code, space padded to 8 characters |
| APVEND | VEND_NM | Vendor name |
| APVEND | ACTV_FLG | Active flag: Y active, N retired |
| APINVH | INV_ID | Invoice identifier |
| APINVH | VEND_CD | Vendor the invoice is from |
| APINVH | INV_NO | Vendor's invoice number |
| APINVH | INV_DT | Invoice date, YYYYMMDD |
| APINVH | INV_AMT | Invoice total |
| APINVH | STAT_CD | Invoice status: O open, P paid, V void |
| APINVH | JRNL_NO | Journal that recorded the invoice |
| APPAYM | PAY_ID | Payment identifier |
| APPAYM | INV_ID | Invoice the payment settles |
| APPAYM | PAY_DT | Payment date, YYYYMMDD |
| APPAYM | PAY_AMT | Payment amount |
