package com.example.llama

/** Pure contract rules shared by the Android transport and executable JVM tests. */
internal object NativeFuturesMetadata {
    fun responseBudget(path: String): Int =
        if (path == "/fapi/v1/exchangeInfo") 4_000_000 else 1_800_000

    fun contractIndex(symbol: String, catalog: List<Pair<String, String>>): Int {
        require(symbol.matches(Regex("[A-Z0-9]{3,22}"))) { "Некорректный символ" }
        val matches = catalog.indices.filter { catalog[it].first == symbol }
        check(matches.size == 1) { "Контракт должен встречаться в каталоге ровно один раз" }
        val index = matches.single()
        check(catalog[index].second == "TRADING") { "Контракт не в торговом статусе Binance" }
        return index
    }
}
