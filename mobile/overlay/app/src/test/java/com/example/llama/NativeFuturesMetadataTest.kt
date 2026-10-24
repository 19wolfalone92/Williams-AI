package com.example.llama

import org.junit.Assert.assertEquals
import org.junit.Assert.fail
import org.junit.Test

class NativeFuturesMetadataTest {
    private fun rejected(block: () -> Unit) {
        try { block(); fail("Invalid contract catalog was accepted") }
        catch (_: IllegalArgumentException) { }
        catch (_: IllegalStateException) { }
    }

    @Test fun selectsRequestedTradingContractFromWholeCatalog() {
        val catalog=listOf("SOLUSDT" to "BREAK", "BTCUSDT" to "TRADING", "ETHUSDT" to "TRADING")
        assertEquals(1,NativeFuturesMetadata.contractIndex("BTCUSDT",catalog))
        assertEquals(2,NativeFuturesMetadata.contractIndex("ETHUSDT",catalog))
    }

    @Test fun absentDuplicateAndNonTradingTargetsAreRejected() {
        rejected { NativeFuturesMetadata.contractIndex("BTCUSDT",emptyList()) }
        rejected { NativeFuturesMetadata.contractIndex("BTCUSDT",listOf("ETHUSDT" to "TRADING")) }
        rejected { NativeFuturesMetadata.contractIndex("BTCUSDT",listOf("BTCUSDT" to "BREAK")) }
        rejected { NativeFuturesMetadata.contractIndex("BTCUSDT",listOf(
            "BTCUSDT" to "TRADING", "BTCUSDT" to "TRADING")) }
        rejected { NativeFuturesMetadata.contractIndex("btcusdt",listOf("BTCUSDT" to "TRADING")) }
    }

    @Test fun onlyExactMetadataRouteGetsLargerBound() {
        assertEquals(4_000_000,NativeFuturesMetadata.responseBudget("/fapi/v1/exchangeInfo"))
        for (route in listOf("/fapi/v1/time","/fapi/v1/klines","/fapi/v1/exchangeInfo/other"))
            assertEquals(1_800_000,NativeFuturesMetadata.responseBudget(route))
    }
}
