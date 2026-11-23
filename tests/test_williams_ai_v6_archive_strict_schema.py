"""Two adversarial archive provenance/schema tests plus positive fixture."""
import unittest
from datetime import date, datetime, timezone
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from williams_ai.v6_official_archive_collector import verified_daily, HOUR_MS

class ArchiveMissingGuards(unittest.TestCase):
    def setUp(self):
        self.date = date(2026,11,11)
        self.symbol = 'BTCUSDT'
        self.name = f'{self.symbol}-1h-{self.date}.zip'

    def payload(self, columns=12):
        start=int(datetime(2026,11,11,tzinfo=timezone.utc).timestamp()*1000)
        out=[]
        for i in range(24):
            ms=start+i*HOUR_MS
            row=[ms, '100', '103', '99', '101', '50', ms+HOUR_MS-1, '0','30','0','0','0']
            out.append(','.join(map(str,row[:columns])))
        mem=BytesIO()
        with ZipFile(mem,'w',ZIP_DEFLATED) as arc:
            arc.writestr(self.name[:-4]+'.csv','\n'.join(out)+'\n')
        return mem.getvalue()

    def fetch(self, data, include_name=True):
        receipt=sha256(data).hexdigest()+ ('  '+self.name if include_name else '')
        def get(url,limit):
            return receipt.encode('ascii') if url.endswith('.CHECKSUM') else data
        return get

    def test_bare_hash_cannot_stand_in_for_filename_bound_checksum(self):
        data=self.payload()
        with self.assertRaisesRegex(ValueError,'INVALID_CHECKSUM_RECORD'):
            verified_daily(self.symbol,self.date,fetch=self.fetch(data,include_name=False))

    def test_signed_truncated_csv_columns_must_fail(self):
        data=self.payload(columns=7)
        with self.assertRaisesRegex(ValueError,'MALFORMED_KLINE_CSV'):
            verified_daily(self.symbol,self.date,fetch=self.fetch(data))

    def test_correct_checksum_and_full_12_column_daily_zip_succeeds(self):
        data=self.payload()
        rows,receipt=verified_daily(self.symbol,self.date,fetch=self.fetch(data))
        self.assertEqual(len(rows),24)
        self.assertTrue(receipt['official_checksum_verified'])

if __name__=='__main__': unittest.main()
