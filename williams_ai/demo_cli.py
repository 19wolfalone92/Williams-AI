"""Interactive DEMO Futures gateway. Never connected to the LLM.

Preview: python -m williams_ai.demo_cli --symbol ETHUSDT --side BUY --quantity 0.2 --stop 95
Submit:  add --send and confirm on TTY with demo-only credentials.

The CLI cannot target production Binance. Never put keys in command line, Git,
screenshots or chat. Does not train AI or accept model-generated orders.
"""
from __future__ import annotations

import argparse
from getpass import getpass
import json
import sys

from .demo_futures import DemoFutures, DemoHttp, DemoSafetyError


def main() -> None:
    parser = argparse.ArgumentParser(description="Williams AI — отдельная Binance Futures DEMO торговля")
    parser.add_argument("--symbol", required=True, choices=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    parser.add_argument("--side", required=True, choices=["BUY", "SELL"])
    parser.add_argument("--quantity", required=True, help="Количество базового актива, не USDT")
    parser.add_argument("--stop", required=True, help="Уровень защитного STOP_MARKET")
    parser.add_argument("--send", action="store_true", help="Разрешить попытку DEMO ордера после подтверждения")
    args = parser.parse_args()

    if args.send and not sys.stdin.isatty():
        parser.error("Для размещения DEMO заявки нужен интерактивный терминал")
    if args.send:
        print("Binance Futures DEMO, НЕ реальный счёт. Создайте отдельные демо API-ключи.")
        key = getpass("DEMO API key (ввод скрыт): ")
        secret = getpass("DEMO API secret (ввод скрыт): ")
        approval = input("Чтобы включить тестовые ордера, напишите DEMO_ACCOUNT_ONLY: ").strip()
        if approval != "DEMO_ACCOUNT_ONLY":
            raise DemoSafetyError("DEMO не разрешена пользователем")
    else:
        key, secret, approval = "PUBLIC_PREVIEW_ONLY", "PUBLIC_PREVIEW_ONLY", ""

    client = DemoHttp(key, secret)
    trader = DemoFutures(client, enabled=args.send, acknowledgement=approval)
    proposal = trader.propose(args.symbol, args.side, args.quantity, args.stop)
    print(json.dumps({
        "mode": "BINANCE_FUTURES_DEMO_PREVIEW",
        "pair": proposal.symbol, "side": proposal.side,
        "quantity": proposal.quantity, "stop_price": proposal.stop_price,
        "max_risk_estimate": proposal.risk_usdt,
        "estimated_notional": proposal.notional_usdt,
        "expires_at": proposal.expires_at,
    }, ensure_ascii=False, indent=2))

    if not args.send:
        print("Только предпросмотр: реальные/демо ордера НЕ выставлялись.")
        return

    confirmation = input("Выполнить ОДИН защищённый DEMO вход? Напишите PLACE_DEMO_ORDER: ").strip()
    if confirmation != "PLACE_DEMO_ORDER":
        print("Отменено. На биржу не отправлялись ордера.")
        return
    result = trader.execute(proposal.ticket, confirmation=confirmation)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Проверьте исполнение позиции и SL непосредственно в Binance DEMO.")


if __name__ == "__main__":
    main()
